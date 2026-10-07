# 文件导读：把逐行 JSON（JSONL）样本适配为训练张量或强化学习对话。
# 预训练读 text；监督微调（Supervised Fine-Tuning，SFT）读 conversations；
# 直接偏好优化（Direct Preference Optimization，DPO）读 chosen/rejected；Agent 另读 gt。
# token 是词表编号，不等于字符；聊天模板把角色、工具声明与轮次边界变为可分词文本。
# input_ids 是上下文，labels 是监督目标；-100 是交叉熵忽略值，不是词表内 token。
# 本文件不移位 SFT 标签，model 的 forward 内移位；DPO 则在数据层显式构造 x/y。
# L=max_length 为补齐后的单样本长度，B 为批大小；DataLoader 堆叠 [L] 为 [B,L]。
# BOS（Begin of Sequence）/EOS（End of Sequence）标记序列边界，PAD（Padding）补齐长度。
# 本词表用聊天消息边界作 BOS/EOS；相同特殊编号在不同任务中承担不同边界职责。
# 先读预训练，再读 SFT 的回答区域扫描，最后比较偏好与在线奖励数据。
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 按 add_system_ratio 为无 system 的对话随机添加系统提示，返回消息列表。
# 有 tools 字段的样本原样返回，避免扰乱工具协议；不原地插入。
# 要求 conversations 非空，否则访问第 0 项失败；概率增强不保证逐次结果一致。
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

# 只处理完全匹配的空 <think> 标签串，返回新字符串。
# 默认保留概率约为 empty_think_ratio；显式布尔参数让偏好对共享处理决定。
# 它不删除非空思考内容，也不是通用 XML 解析器。
def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# 无条件文本预训练适配器；每行至少有 text，输出定长输入及标签。
# 目标是下一 token 预测，不含“只监督回答”的角色区分。
class PretrainDataset(Dataset):
    # 保存 tokenizer 与 max_length，交给 datasets 读取 JSON 文件的 train 切分。
    # 此处 train 是加载切分名，不表示已自动拆出验证集。
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    # 返回样本条数，供采样器计算索引范围；不返回 token 总数。
    def __len__(self):
        return len(self.samples)

    # index 取一条 text，截至 max_length-2，为起止特殊 token 留两格。
    # 返回两个 [max_length] long 张量；pad 对应标签被置 -100，不参与损失。
    # labels 克隆后修改，避免把 input_ids 中的合法编号也改成 -100。
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # 填充值通常是专用 pad 编号；按编号匹配会忽略所有相同编号的位置。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


# 对话监督适配器：整段对话作为输入，只为 assistant 内容及结束边界生成标签。
# 用户/system/tool 文本仍供注意力读取，但不直接作为要预测的监督目标。
class SFTDataset(Dataset):
    # 声明 conversations 字段类型，存储分词器和长度上限。
    # bos_id/eos_id 是多 token 边界序列，不能等同一个整数 BOS/EOS。
    # tools/tool_calls 在该加载 schema 中按字符串读入，后面再解析 JSON。
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    # 返回对话样本数，每条可含多轮；训练批量按对话条数计算。
    def __len__(self):
        return len(self.samples)

    # 拷贝消息字典，解析 system.tools 与字符串 tool_calls 后交给聊天模板。
    # 返回文本（tokenize=False），无新增待生成前缀，因为样本含完整回答。
    # 模板负责角色序列化；本方法不拼接自定义标记，也不更新模型参数。
    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    # input_ids 为整数列表；返回等长列表，默认全 -100。
    # 扫描 assistant 起始序列后，将回答至 eos_id（含该边界）设为原编号。
    # 若截断使结束标记缺失，就监督到现有尾部；这是序列匹配，不解析真实角色结构。
    # 此函数不额外排除 PAD：未找到结束标记且输入尾部已有填充时，填充值也会被标为目标。
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 切片比较完整起始序列，不是检查单一 BOS；随后从角色前缀之后监督。
            # 扫描尾部切片不足长度时自然不匹配；正文若含同样边界 token，也可能触发。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    # 依次随机补 system、套模板、随机删空思考、分词截断与填充。
    # 返回 input_ids/labels 两个 [L] 张量；本接口未返回 attention_mask。
    # 因此标签忽略与注意力屏蔽不是一回事，模型能否看 pad 由调用方决定。
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# 偏好对数据：chosen 为较好回答，rejected 为较差回答。
# 两条完整对话分别分词，输出 x/y 及 0/1 回答遮罩供 train_dpo 比较概率。
class DPODataset(Dataset):
    # 初始化长度及聊天边界序列，加载偏好 JSONL。
    # self.padding 保存了回退值，但实际 padding 由 tokenizer(...,padding=...) 完成。
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    # 返回偏好对数量；一个索引对应两条序列，不是两条独立样本。
    def __len__(self):
        return len(self.samples)

    # 同时渲染 chosen/rejected，共享空思考删除决定，避免随机格式破坏可比性。
    # 返回字典中 x/y/mask 均为 [L-1]；偏好对须自行保证同一问题背景。
    # 无梯度，仅构造数据；训练侧先拼 chosen 后拼 rejected，损失据此分半。
    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        remove_empty_think = random.random() > 0.2
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 若原 token 为 [a,b,c,d]，x=[a,b,c]，y=[b,c,d]。
        # mask 同样取 [1:]，必须跟 y 而非 x 对齐；此路径不再传 labels 让模型二次移位。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    # 返回 [L] 的 0/1 列表，扫描规则与 SFT 回答区域相同。
    # 1 覆盖回答及结束边界；它参与对数概率聚合，不是模型 attention_mask。
    # 不另按 PAD 编号屏蔽；未找到完整 eos_id 时可能把尾部填充也纳入回答区域。
    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 切片比较完整起始序列，不是检查单一 BOS；随后从角色前缀之后监督。
            # 扫描尾部切片不足长度时自然不匹配；正文若含同样边界 token，也可能触发。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# AI 反馈强化学习（Reinforcement Learning from AI Feedback）提示数据。
# 只返回 prompt/answer 字符串，答案由策略现场采样、由奖励模型评分。
class RLAIFDataset(Dataset):
    # 加载 conversations，保存 thinking_ratio 控制是否打开思考前缀。
    # max_length 在此仅存储；训练脚本另在分词时或分词后限制提示长度。
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    # 返回提示样本数，独立于每个提示采样多少个回答。
    def __len__(self):
        return len(self.samples)

    # 去掉 conversations 最后一条，再加 assistant 待生成前缀。
    # 约定最后一条是原回答；若输入并非此格式，也会照样删尾。
    # 按概率设置 open_thinking，影响模板前缀，不保证模型一定正确推理。
    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    # 返回渲染提示及空 answer；不是用原最后一条作为奖励真值。
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

# 多轮工具强化学习样本，保留结构化 messages/tools 及可核对答案 gt。
# 不用默认张量批处理，由 train_agent 的 collate_fn 维持嵌套列表。
class AgentRLDataset(Dataset):
    # 逐行 json.loads 到内存；max_length 仅记录，未执行截断。
    # 空行或损坏 JSON 会抛异常，不能解释为自动跳过无效样本。
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    # 返回内存中的 JSONL 行样本数。
    def __len__(self):
        return len(self.samples)

    # 拷贝消息并解析 system.tools，返回 (去掉最后一条的消息,工具表)。
    # 尾消息按数据约定是示范答案；tools 未出现时为 None。
    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    # 返回 messages、tools、gt 字典；缺少 gt 会直接 KeyError。
    # 上游需准备完整字段，下游据 gt 与工具执行轨迹算奖励。
    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


# 此入口只有 pass；直接执行并不下载数据或启动训练。
if __name__ == "__main__":
    pass
