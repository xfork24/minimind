# 文件导读：仓库根目录的命令行对话入口，支持原生权重、LoRA 与 Transformers 模型。
# 输入人工问题或示例问题；输出流式文本及可选端到端 token/s，不做训练。
# 运行须准备 tokenizer 和匹配权重；historys 按消息条数截取，通常问答两条为一轮。
# 阅读 main 的路径配置/历史/模板，再读 init_model；生成原理见 model/model_minimind.py。
# 自回归（Autoregressive）逐 token 预测，每步使用提示及已生成内容。
# token 是分词器编号；B=批大小，P=提示 token 数，R=新增 token 数，完整输出为 [B,P+R]。
# LoRA（Low-Rank Adaptation，低秩适配）训练增量矩阵，MoE（Mixture of Experts，混合专家）选少量前馈专家。
# RoPE（Rotary Position Embedding，旋转位置编码）给注意力注入位置；改变位置表不保证长文本质量。
# temperature 以 logits/温度调分布尖锐程度，top-p 保留累计概率达到阈值的候选再采样。
# eval 关闭训练态随机失活，no_grad/inference_mode 关闭梯度记录；二者分别控制不同事情。
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# 按 args.load_from 加载 tokenizer 与模型，返回 (半精度、eval、指定设备模型, tokenizer)。
# 路径包含字符串 model 时走原生 .pth 分支（不是精确目录名判断），结构须匹配权重。
# 否则走 Transformers 自动类，trust_remote_code=True 可执行模型目录中的自定义代码。
# eval 关闭 dropout 等训练行为，不等于 no_grad；此处统一 half，CPU 也走半精度。
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 原生分支由 CLI 构造 config，再 strict=True 加载；层数/宽度/MoE 或词表不匹配会报错。
    # LoRA 若启用，需先注入同结构再加载 A/B；不是读取适配器就能替代基模。
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

# 解析 CLI、加载模型，循环生成并追加 assistant 回答；返回 None，打印输出。
# 预训练模式拼 BOS+裸问题，对话模式套模板；随机种子每问重新随机选择。
# 手动输入空串结束；参数与输出目录以仓库根目录为基准。
def main():
    # 选项控制起始权重、模型形状、思考开关、历史和采样；不会自动寻找缺失数据/权重。
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # iter(callable, sentinel) 每次调用 input，遇空串停止；自动模式遍历固定列表。
    # 这里的 lambda 无参数，只读一条问题，不做模型推理。
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # historys=0 显式清空；否则保留末尾 N 条消息（不是 N 轮），再追加当前问题。
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            # 聊天模板（Chat Template）把 messages 的角色与边界转为模型训练过的文本协议。
            # add_generation_prompt 为待生成助手加前缀；tools 只声明工具，不在模型内部执行函数。
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        # 输入 [1,P]；max_new_tokens 限新增步数，返回 [1,P+R]，包含原提示。
        # 温度缩放改变尖锐程度，top_p 保留累积概率阈值内候选再采样；具体本地算法见 generate。
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 按提示 token 数切掉前 P 个 ID，避免把问题再次当回答；跳过特殊 token 后入历史。
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        # 分母从 generate 前开始，含提示计算/采样/流式输出开销，不是纯 GPU decode 内核基准。
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    try:
        main()
    # 仅捕获用户结束输入/中断，模型加载与运行异常仍传播。
    except (EOFError, KeyboardInterrupt):
        print()