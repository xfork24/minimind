# 文件导读：实现 MiniMind 自回归语言模型：根据已有 token 预测下一个 token。
# token 是分词器映射出的离散编号，不等于一个汉字或一个单词；词表大小决定输出类别数。
# 上游是 dataset 的输入编号，下游是 trainer 的损失优化及 eval/scripts 的逐步生成。
# 张量约定：B=批量，T=当前序列长度，D=隐藏维度，V=词表大小，H=查询头数，K=键值头数，d=头维度。
# 主路径为编号 [B,T] → 词嵌入 [B,T,D] → 多层注意力/前馈 → logits [B,T,V]。
# logits 是未归一化分数；softmax 才把它们变成词表上的概率分布。
# softmax(z)_i=exp(z_i)/sum_j exp(z_j)，分母沿词表轴求和；各 token 类别概率之和为 1。
# Transformer 用注意力混合上下文，再用逐位置前馈变换特征；层数与宽度决定表示容量。
# 建议从 MiniMindConfig、RMSNorm、Attention 读到 MiniMindBlock，再读 forward 与 generate。
# 依赖 PyTorch 与 Transformers；配置、分词器及权重形状必须一致，本文件不自行加载训练数据。
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
# 配置对象沿用 Transformers 的序列化约定，保存架构及生成用的特殊 token 编号。
# 同一组权重必须配同一隐藏宽度、层数、词表与专家结构；配置不是训练超参数的自动校验器。
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    # hidden_size 决定 D，num_hidden_layers 决定深度，use_moe 选择稠密或混合专家前馈。
    # kwargs 既传给基类，也用于补充头数、位置编码、激活等选项；构造过程只存配置。
    # 注意 head_dim 应为偶数，H 应能被 K 整除；本实现不显式检查这些前提。
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # hidden_size：隐藏层宽度 D，所有子层输入/输出向量的统一维度，决定模型体格。
        # num_hidden_layers：Transformer 块堆叠层数，越深表达越强、训练越慢。
        # use_moe：是否把普通 FFN 换成 MoE（混合专家）层，见下方 MoE 配置。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        # dropout（随机失活）在训练中随机置零部分激活并缩放保留值，降低过拟合。
        # eval 模式关掉模块的随机失活；默认概率 0 表示本模型通常无此随机性。
        # vocab_size：词表大小，embedding 矩阵行数 = lm_head 输出维度，决定能表示多少个 token。
        # bos/eos_token_id：起始/终止符编号；编码时以 <s> 开头，生成时采样到 </s> 即停止。
        # flash_attn：True 且条件满足时走 PyTorch SDPA（FlashAttention 内核），更快更省显存；否则退回手写注意力。
        # num_attention_heads：查询头数量 Q；num_key_value_heads：KV 头数量 K，K<Q 即 GQA，多个查询头共享一组 K/V。
        # head_dim：每头宽度 = hidden_size / Q（见上：须为偶数且 Q 能被 K 整除）。
        # hidden_act：FFN 激活函数，silu 即 SwiGLU 门控用的 SiLU（x·sigmoid(x)）。
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # 中间宽度取约 pi*D 并向上对齐 64，便于矩阵运算；不是序列最大长度。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # max_position_embeddings：RoPE 预计算表覆盖的最大位置数，即模型支持的最大上下文长度。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # rms_norm_eps：RMSNorm 分母 sqrt(mean(x²)+eps) 里的稳定项，防除零；太大偏移归一化尺度。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # rope_theta：RoPE 基频 base（θ_i = base^(-2i/d)），越大低频分量越多，长距离建模与外推越强。
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # tie_word_embeddings：True 时输入 embedding 与输出 lm_head 共享同一权重矩阵，省 vocab*D 参数。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # inference_rope_scaling：YaRN 外推开关，True 时下方 rope_scaling 生效（2048 训练长度外推到 32k）。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        # YaRN（Yet another RoPE extensioN）按频段混合原频率与缩小后的频率。
        # 此处 factor=16、original_max=2048；以实际配置为准，不把命令行帮助的倍数当实现。
        # 原始论文：https://arxiv.org/abs/2309.00071
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        # 专家数是总容量，num_experts_per_tok 是每 token 的活跃专家数。
        # router_aux_loss_coef 仅调均衡项权重，不是优化器学习率。
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        # moe_intermediate_size：每个专家 FFN 的中间宽度，默认与稠密 intermediate_size 一致。
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # norm_topk_prob：True 时把 Top-K 门控权重归一化（和为 1），专家输出按权重比例合并。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
# 均方根归一化（Root Mean Square Normalization）稳定每个 token 的特征幅度。
# 与 LayerNorm 不同，它不减均值；只有可学习的逐通道缩放，没有平移偏置。
class RMSNorm(torch.nn.Module):
    # dim 是最后一维宽度；eps 防止零向量除零，weight 初始为全 1。
    # weight 是 Parameter，会进入模型参数表，并在训练时接收梯度。
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # 输入任意前导维度、末维为 dim；返回相同形状。
    # 公式为 x / sqrt(mean(x²)+eps)；keepdim=True 保留末维 1，使分母广播到各通道。
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 先转 float32 算平方与归一化，减少低精度溢出，再乘缩放并恢复输入 dtype。
    # 类型转换仍保留梯度链路；这不是 detach，也不会冻结权重。
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

# 预计算旋转位置编码（Rotary Position Embedding，RoPE）的余弦与正弦表。
# dim=d，end=支持的位置数；返回两张 [end,d] 表，供 Q/K 按位置旋转。
# 每对通道频率为 theta^(-2i/d)，位置 p 对应角度 p*频率；不直接加位置向量。
# rope_scaling 可启用此处的 YaRN 分频插值；改变位置表不保证模型具备对应长文本能力。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # 反解“原窗口内转了 b 圈”对应的频率下标，确定插值渐变区间。
            # 匿名函数输入 b，输出浮点下标；low/high 截至可用通道范围。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp=[d/2] 从 0 线性增至 1；低端保留原频率，高端除 factor。
            # max(high-low,0.001) 防止渐变区间退化时除零。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    # 外积将 [end] 位置乘 [d/2] 频率得 [end,d/2] 角度表。
    # 拼两份 cos/sin → [end,d]，配合“前半/后半”旋转通道约定。
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 给 q=[B,T,H,d]、k=[B,T,K,d] 注入位置信息，返回同形状的旋转结果。
# 表切片是 [T,d]，默认 unsqueeze(1) → [T,1,d]，隐式补批量轴后广播到各头。
# 旋转只作用于 Q/K，不作用于 V；无参数更新，但可反向传播至投影层。
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 把最后一维分为两半，拼成 [-后半,前半]，实现二维旋转中的正交分量。
    # 例如 [a,b,c,d] → [-c,-d,a,b]，配合重复的 cos/sin 完成两两配对。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

# 分组查询注意力（Grouped Query Attention，GQA）让多个 Q 头共享一组 K/V。
# 输入 [B,T,K,d]，输出 [B,T,K*n_rep,d]；n_rep=1 时原样返回。
# 例如 H=8、K=4 时，每个 K/V 头复制两次，匹配 8 个查询头而减少缓存大小。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    # 插入复制轴 [B,T,K,1,d]，expand 成 [B,T,K,n_rep,d]，再合并头轴。
    # expand 用步长视图广播；随后的 reshape 可能分配实际存储，不宜宣称全程零拷贝。
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

# 因果自注意力：Q（查询）寻找相关位置，K（键）供匹配，V（值）提供聚合内容。
# 每个 token 只能看自身和过去；多头并行捕捉不同关系，输出仍为 [B,T,D]。
# 缓存（KV cache）复用历史 K/V，减少逐 token 推理的重复投影；训练通常不缓存。
class Attention(nn.Module):
    # 按 config 建立无偏置 Q/K/V/O 投影、每头 Q/K 归一化和 dropout。
    # flash 表示允许调用 PyTorch SDPA（缩放点积注意力），不保证一定使用 Flash 内核。
    # 实际内核由设备、dtype、形状及 PyTorch 决定；手工路径作为带遮罩/缓存的回退。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    # x=[B,T,D]，position_embeddings 是当前位置的 cos/sin；past_key_value 为历史 (K,V)。
    # 返回 (output,past_kv)，后者只在 use_cache=True 时给出，按时间轴累计未复制的 K/V。
    # attention_mask 通常为 [B,历史长度+T]，1=可见、0=填充；不是损失标签遮罩。
    # Q/K/V 投影与注意力聚合可微；传入缓存时调用者须保证长度、设备与批量一致。
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 先沿 D 投影，再 view 拆出头维；例如 [2,10,768] 的 Q 可变为 [2,10,8,96]。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            # 历史键与当前键沿时间轴拼接；值同理，缓存不保存 Q。
            # 先旋转当前 K 再拼历史 K，避免对历史位置重复旋转。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 缓存仍是 [B,L,K,d]，计算时才重复 K/V 并转成 [B,H,L,d]。
        # L=历史长度+T；Q 转为 [B,H,T,d]，因此分数为 [B,H,T,L]。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 仅无历史多 token、无实际 padding 等条件满足时调用 SDPA。
        # 显式传 eval 时 dropout_p=0；SDPA 不会仅因模块 eval 就自动关掉传入的 dropout。
        # 接口核对：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # Q*Kᵀ 计算匹配分数，除 sqrt(d) 防止随头宽度增大而 softmax 过于尖锐。
            # 转 float32 是在矩阵乘后，不能消除此前低精度乘法的全部误差。
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            # 只在最后 T 个键列加 [T,T] 上三角 -inf；历史列全部允许访问。
            # 例如缓存 5、当前 2：第一个新查询不可见第二个新 token，但可见 5 个历史 token。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # [B,L] → [B,1,1,L] 广播至所有查询头；padding 列加 -1e9。
            # 这是可见性遮罩，不是 -100 标签；整行全被屏蔽的异常输入未单独处理。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        # [B,H,T,d] 转回 [B,T,H,d] 再合并 H*d，O 投影回 D。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# 门控前馈网络独立处理每个位置，把 D 扩展到中间宽度再压回 D。
# 默认 SiLU 激活构成 SwiGLU 风格门控；它补充非线性表达，不混合时间位置。
# SiLU（Sigmoid Linear Unit）为 x*sigmoid(x)，sigmoid(x)=1/(1+exp(-x))；
# SwiGLU 是带 SiLU 的门控线性单元，两个投影逐元素相乘，而非注意力概率相乘。
class FeedForward(nn.Module):
    # intermediate_size 可覆盖配置宽度；gate_proj、up_proj 扩维，down_proj 降维。
    # 权重由 PyTorch/模型初始化机制管理；无前向缓存，也不在构造时计算损失。
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    # 输入输出同为 [...,D]：down(act(gate(x))*up(x))，* 是逐元素乘法。
    # 门值不是硬开关，梯度沿两条投影分支传播；中间维度必须一致才能相乘。
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# 混合专家（Mixture of Experts，MoE）为 token 选择少量前馈子网络。
# 总参数量可大于每个 token 的活跃参数量；路由器输出专家概率，并记录负载均衡辅助损失。
class MOEFeedForward(nn.Module):
    # 创建 D→专家数的 gate 及独立 FeedForward 专家列表。
    # ModuleList 保证所有专家被注册、保存与优化；此实现没有共享专家。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    # 输入 [B,T,D] 展平为 [B*T,D]，每个 token 路由到 top-k 专家再加权汇合。
    # 返回 [B,T,D]，并更新 self.aux_loss，外层汇总它后交由训练脚本加入总损失。
    # 离散 top-k 索引不可微；选中的权重及专家输出可微，不应把选专家解释成可微采样。
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # scores=[B*T,E]；top-k 得权重/索引 [B*T,k]。
        # 只运行被选专家；k 必须在 1..E 内。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1 的直通估计：top1-top1.detach()+1 前向恒为 1，反向仍流向 gate。
                # 此分支重新用 x_flat.detach() 算概率，路由权重梯度不回传输入，专家分支仍回传。
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                # token_idx 找出派给专家 i 的行，weight=[命中数,1] 广播乘专家输出。
                # index_add_ 原地累加多专家贡献；未命中分支乘 0 挂参数梯度链，数值不变。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # one_hot=[B*T,k,E]，mean(0) 得 [k,E] 路由频率。
            # 与 scores.mean(0) 的 [E] 广播相乘并全求和；k>1 时这里对 k 轴求和而非取均值。
            # 辅助项鼓励使用分散，离散 load 本身不回传梯度。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

# 预归一化（Pre-Norm）Transformer 块：注意力与前馈各有一次残差相加。
# 残差提供短梯度路径，子层只学习在已有表示上增加的改变量。
class MiniMindBlock(nn.Module):
    # 按 config 创建两种归一化及注意力/前馈；layer_id 传入但此处未用于计算。
    # use_moe 只替换前馈路径，不替换注意力。
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    # hidden_states=[B,T,D]；位置表与该层缓存传给注意力。
    # 返回新隐藏状态及该层缓存；两次残差不改变形状，训练梯度穿过整块。
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 第一处 += 为原地残差加法；下一处 + 建立新结果。
        # 两条路径都参与反向传播，不能把残差理解为停止梯度。
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

# 主干网络将 token 编号转换为上下文化向量，不直接输出词表概率。
# 组织词嵌入、多个块与最终 RMSNorm；辅助损失和缓存向上层语言模型透传。
class MiniMindModel(nn.Module):
    # 注册词嵌入 [V,D] 及层列表；预计算 RoPE 表作为 buffer 随模型迁移设备。
    # persistent=False 表示位置表不进入 state_dict；它由配置重建，不是可训练参数。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # input_ids=[B,T]；返回 (hidden_states,presents,aux_loss)。
    # 从第一层历史 K 的时间长度取 start_pos，使新增 token 用正确绝对位置。
    # 带 layers 属性的 Transformers 缓存被丢弃，不是转换成原生缓存。
    # 原生缓存列表应覆盖全部层；zip 只迭代较短一方，不会自动补齐缺层。
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 词嵌入（Embedding）按整数 ID 查 [V,D] 参数表，得到每个位置的 D 维向量。
        # 梯度流向查到的参数行；ID 本身是离散索引，不对编号求导。
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # 正常首位置 cos=1；若初始化产生全零表，则据配置重算并迁移至输入设备。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# 因果语言模型（Causal LM）在主干上增加词表投影及生成接口。
# 训练损失是下一个 token 的交叉熵；辅助损失单独返回，由 trainer 决定是否相加。
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # config=None 时用默认配置；输出头 D→V，与词嵌入可共用同一个 Parameter。
    # 权重共享（weight tying）减少参数，并让输入/输出 token 表示共同优化。
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    # 返回 Transformers 风格对象，含 loss、logits、缓存、隐藏状态及 aux_loss。
    # logits_to_keep=0 保留全部位置；正整数只保留尾部，张量可作索引。
    # labels 模式依赖完整、对齐的 logits；仅截尾 logits 又传全长 labels 会形状不匹配。
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # Python -0 等于 0，所以默认 slice(0,None) 保留全序列，而非“保留零个”。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 预测/目标错开一位：[a,b,c] 用 a 处 logits 监督 b、b 处监督 c。
            # contiguous 后展平成 [B*(T-1),V] 与 [B*(T-1)]；-100 不计交叉熵。
            # 若整批目标均被忽略，均值损失可能为 NaN；本层没有额外兜底。
            # 例如目标概率从 0.1 提升到 0.5，单位置损失从约 2.303 降到 0.693（自然对数）。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    # 无梯度自回归生成：反复前向、筛选词表、取一个 token、追加至上下文。
    # inputs 或 kwargs['input_ids'] 提供 [B,P] 提示；默认返回含提示的完整编号序列。
    # return_kv=True 改为字典；streamer 接收 CPU 编号，所有样本遇 EOS 或到长度上限才结束。
    # 温度必须大于 0、top_k 不应超过词表宽度；此入口未统一校验所有参数。
    # 缓存对应最后一次前向，通常尚未包含刚采出的最后一个 token，续写时需留意边界。
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # 输入与缓存独立传入；缓存不是生成 token 文本，也不替代完整上下文记录。
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 只把缓存之后的新 token 送入前向；关闭缓存时每轮重新计算完整序列。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 尾位置预测下一 token；temperature<1 使分布尖锐，>1 更平坦。
            # logits 未归一化，不能直接当概率。
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # 重复惩罚按符号缩放已出现 token 的分数：正数除 penalty，负数乘 penalty。
                    # penalty>1 时压低分数，0<penalty<1 时反而鼓励重复；penalty=1 不改变分数。
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                # top-k 仅保留分数最高的 k 个候选（阈值并列项也可能留下），其余置 -inf。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 核采样（top-p）按概率降序累计；右移遮罩保留第一个跨越阈值的 token。
                # clone 防止重叠切片原地赋值问题；scatter 将排序空间遮罩映射回词表编号。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 采样按概率随机抽取，贪心取 argmax；两者都只生成一个新编号。
            # 已结束样本继续填 EOS，使批量形状一致；finished 的 |= 是原地累计结束状态。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids