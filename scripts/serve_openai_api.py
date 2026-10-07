# 文件导读：FastAPI 包装本地 MiniMind 生成，提供 /v1/chat/completions 兼容接口。
# 输入消息/采样参数/工具声明，输出完整 JSON 或 SSE（Server-Sent Events，服务端事件流）。
# 宿主只生成 tool_calls，不执行工具；调用方需要自行执行并带结果再次请求。
# 从 scripts 目录启动，准备匹配权重；全局 model/tokenizer 在 __main__ 初始化。
# 先读 ChatRequest→chat_completions→流式线程/队列；示例未提供生产认证或并发资源调度。
# token 是分词器编号；B=批大小，P=提示 token 数，R=新增 token 数，完整输出为 [B,P+R]。
# LoRA（Low-Rank Adaptation，低秩适配）训练增量矩阵，MoE（Mixture of Experts，混合专家）选少量前馈专家。
# RoPE（Rotary Position Embedding，旋转位置编码）给注意力注入位置；改变位置表不保证长文本质量。
# temperature 以 logits/温度调分布尖锐程度，top-p 保留累计概率达到阈值的候选再采样。
# eval 关闭训练态随机失活，no_grad/inference_mode 关闭梯度记录；二者分别控制不同事情。
import argparse
import json
import re
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


# 按 args.load_from 加载 tokenizer 与模型，返回 (半精度、eval、指定设备模型, tokenizer)。
# 路径包含字符串 model 时走原生 .pth 分支（不是精确目录名判断），结构须匹配权重。
# 否则走 Transformers 自动类，trust_remote_code=True 可执行模型目录中的自定义代码。
# eval 关闭 dropout 等训练行为，不等于 no_grad；此处统一 half，CPU 也走半精度。
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 原生分支由 CLI 构造 config，再 strict=True 加载；层数/宽度/MoE 或词表不匹配会报错。
    # LoRA 若启用，需先注入同结构再加载 A/B；不是读取适配器就能替代基模。
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.half().eval().to(device), tokenizer


# Pydantic 请求模型，对基础字段解析与默认值赋值；messages/tools 内部仅声明为 list。
# 未限制温度范围、消息角色 schema 或最大并发；model 名字不会在这里动态加载不同权重。
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    # 优先显式 open_thinking，再兼容模板 kwargs 的 open_thinking/enable_thinking。
    # 返回开关值，不修改模型；False 不会否定 kwargs 中的 True。
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


# 继承 TextStreamer，把完成的文本片段送到线程安全队列，让 HTTP 消费者逐段读取。
# 跳过原提示与特殊 token；流式文本片段不一定是一整个 token。
class CustomStreamer(TextStreamer):
    # 初始化父类解码行为，保存 tokenizer 与调用方 Queue；无独立模型。
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    # 回调把 text 入队，结束时再入 None 哨兵；消费者据此结束，不把 None 当文本。
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


# 分离思考区、普通正文与工具调用，返回 (content,reasoning_content,tool_calls或None)。
# 无起始 think 但有闭合标签时把前段当思考；工具参数编码成 API JSON 字符串。
# 正则/JSON 解析失败忽略对应调用，不验证工具允许列表。
def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


# 生成器渲染 prompt 并开生成线程，从队列读取增量，yield JSON 字符串。
# emitted 是已发送字符偏移，full_text 是累计文本；最后集中解析工具调用并发 finish_reason。
# 思考开关影响增量分类；异常以 error JSON 输出，不直接更新 HTTP 状态码。
def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        # 聊天模板（Chat Template）把 messages 的角色与边界转为模型训练过的文本协议。
        # add_generation_prompt 为待生成助手加前缀；tools 只声明工具，不在模型内部执行函数。
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 生产者为 generate 线程，消费者为 SSE 迭代器；队列无容量上限，慢客户端可能积压。
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        # 闭包读取 inputs/model/queue，线程内调用 generate；异常入 error 再入结束哨兵。
        # 无梯度优化；线程生成与 HTTP 读取解耦，但未实现客户端断连的取消控制。
        def _generate():
            try:
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except Exception as e:
                queue.put({"error": str(e)})
                queue.put(None)

        Thread(target=_generate).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            full_text += text

            if not thinking_ended:
                # 寻找完整 </think> 后切换字段；检测前会立即发送所有未发字符。
                # 结束标签若跨片段到达，前半标签可能已作为思考增量发出，非完整标签缓存解析器。
                pos = full_text.find('</think>')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        # 普通文本可能已经流出工具标签；结束时再附结构化调用，不等于只发送其中一种格式。
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


# POST 处理器读取请求，流式返回 StreamingResponse，非流式同步生成完整回答。
# 非流式 no_grad，切掉提示 IDs，再拆 thinking/tools；异常转 HTTPException。
# async 声明不意味着同步模型计算自动不阻塞事件循环；此例未做并发隔离。
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                # SSE 每事件以 data: 开头、空行结尾；这里发送 finish_reason，但未另发送 [DONE] 哨兵。
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            # 聊天模板（Chat Template）把 messages 的角色与边界转为模型训练过的文本协议。
            # add_generation_prompt 为待生成助手加前缀；tools 只声明工具，不在模型内部执行函数。
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    # 监听所有网卡的 8998 端口；示例无身份认证，部署前须自行限制暴露与请求资源。
    uvicorn.run(app, host="0.0.0.0", port=8998)
