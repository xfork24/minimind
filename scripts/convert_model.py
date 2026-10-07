# 文件导读：在原生 PyTorch state_dict 与 Transformers 模型目录间转换，另提供 LoRA 合并。
# 输入权重与全局 lm_config，输出模型/tokenizer/config；不包含训练优化器历史。
# 从 scripts 目录运行，默认主入口调用 Qwen 兼容导出，其他示例仍被注释。
# 先确认 hidden_size/层数/词表/MoE 与输入一致，再读对应转换函数。
# state_dict 是“参数名→张量”的映射；文件格式兼容不自动保证模型数学实现完全一致。
# LoRA（Low-Rank Adaptation，低秩适配）用两小矩阵表示增量，MoE（Mixture of Experts，混合专家）分路由专家。
# RoPE（Rotary Position Embedding，旋转位置编码）和激活函数属于模型计算约定，转换不能仅检查参数名字。
import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)

# 载入原生参数并注册 Auto 类，导出保留 MiniMind 自定义代码的 Transformers 目录。
# dtype 控制导出精度；strict=False 允许未匹配 key，调用者必须自行审视兼容性。
# 读取全局 lm_config 与 ../model tokenizer，写目标目录并打印参数量，返回 None。
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    # 注册 auto_map，使自动类能找到随目录保存的自定义配置/模型源码。
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # ======= transformers-5.0的兼容低版本写法 =======
    # 按库主版本处理导出配置兼容：调整 tokenizer 字段并转换 RoPE 配置形式。
    # 代码假定 rope_parameters 存在，del 缺 key 会失败；此处只解释，不修补原逻辑。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
# 用全局 lm_config 构造 Qwen3 稠密或 MoE 兼容结构，重排必要的专家权重并 strict=True 加载。
# 输出标准 Transformers 权重与分词器；依赖当前库版本结构，不重新训练参数。
# 结构相似并非对全部配置的通用转换器，调用者须保持实际层/激活/RoPE 配置一致。
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    # 把共享尺寸映射到 Qwen 配置；embedding/head 权重绑定规则也须兼容。
    # MoE 额外设置专家数与每 token 选专家数，不改变训练好的专家值。
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False, 
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        # 此分支只适配 Transformers 5.x 的 MoE 参数布局；稍后才修改导出的配置文件。
        # 版本分支假定对应 Qwen3Moe 实现使用下列张量键，实际兼容性仍由严格加载检查。
        if int(transformers.__version__.split('.')[0]) >= 5:
            # 新版专家由逐专家 Linear 转为整块张量：gate/up 各 [E,I,D]，沿 dim=1 拼 [E,2I,D]。
            # down 堆叠为 [E,D,I]；E=专家数、I=中间宽度、D=隐藏宽度。原逐专家 key 被移除。
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0的兼容低版本写法 =======
    # 按库主版本处理导出配置兼容：调整 tokenizer 字段并转换 RoPE 配置形式。
    # 代码假定 rope_parameters 存在，del 缺 key 会失败；此处只解释，不修补原逻辑。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


# 加载 Transformers 模型，导出 CPU float16 state_dict 到 torch_path，返回 None。
# 不重命名参数、不重建训练状态；目标原生结构仍须与导出 key 匹配。
def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


# 构造基模并注入 LoRA，委托 merge_lora 导出 W+BA 的普通结构权重。
# 输入三条路径，输出落盘；不启动训练，使用全局 config 决定参数形状。
def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


# 读取模板文本并 json.dumps 转义，再打印 chat_template 字段片段。
# 返回 None；打印的引号/反斜杠是 JSON 编码，不是改变模板运行含义。
def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


# 从配置 JSON 提取 chat_template 原字符串写为模板文件，缺字段则报错。
# 只恢复转义后的文本，不渲染 messages，也不执行模型。
def convert_json_to_jinja(json_file_path, output_path):
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


# 默认只执行 Qwen 兼容导出；修改示例前需确认输入路径和 config 与已有权重一致。
if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
