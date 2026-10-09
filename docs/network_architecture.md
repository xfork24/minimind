# MiniMind 完整网络结构图（Embedding → ×N Block → lm_head）

下图按 `model/model_minimind.py` 的**实际网络结构**绘制, 分三节:

- **① 整体结构**（`MiniMindForCausalLM` 前向）: embedding → dropout → ×N 层 Block → 最终 RMSNorm → lm_head → logits → softmax/采样
- **② Attention 内部**: 投影 / per-head RMSNorm / RoPE / KV 缓存 / 多头 GQA 分组 / 注意力核心七步
- **③ FeedForward 内部**: SwiGLU（gate/up/SiLU/⊗/down）
- **④ MoE 替换**（`use_moe=True` 时替换 ③ 的 mlp）: gate → softmax → top-1 → 专家 → Σ 累加

> 全图连线均为**正交折线**（走行间通道或侧向旁路, 不穿过任何组件）; 连线层位于组件层之下; 各组件互不遮挡（HTML 版已程序化验证 63 个节点零重叠）。

> 每个网络结构只画**一个浅灰平行四边形块**（模块类型、参数矩阵、计算方式展开写在块内）; 矩形 = 激活值张量; 圆 = 逐元素算子（⊕ 残差 / × 门控 / Σ 累加）; 不使用决策菱形。图中公式为 mathtext（Computer Modern）排版, 与正文 LaTeX 符号一致。

![MiniMind 完整网络结构图](./network_architecture.svg)

- **静态矢量图**: [network_architecture.svg](./network_architecture.svg)（文本为字形路径, 任何查看器都不会重排换行）
- **可拖动交互版**: [network_architecture.html](./network_architecture.html)（浏览器打开; 拖动模块调整布局、连线自动跟随; 拖动**连线中点的小圆点**可手动调整该线走线通道（保持正交）, 双击圆点恢复自动走线; 公式由 KaTeX 渲染, 需联网）

重新生成（改 `scripts/gen_network_diagram.py` 中的 `NODES` / `EDGES` / `SECTIONS` 后）:

```bash
uv pip install --python .venv/bin/python matplotlib   # 首次需要
.venv/bin/python scripts/gen_network_diagram.py
```

## 图形约定

| 形状 / 线型 | 含义 |
|---|---|
| 平行四边形（浅灰底） | 网络结构块: 一个模块一个块, 块内展开写明类型、参数矩阵形状与计算 |
| 矩形（白底） | 激活值（张量）, 标注完整维度 |
| 圆 | 逐元素算子: ⊕ 残差相加 / × 门控逐元素乘 / Σ MoE 加权累加 |
| 实线箭头 | 前向数据流（恒定执行的主结构） |
| 虚线箭头 | 条件路径: 推理期 KV 缓存、可选输入 `attention_mask`、`×N` 层循环、MoE 替换（`use_moe=True`）、generate 阶段采样 |
| 虚线圆角框 | 重复单元容器: `MiniMindBlock × N` |
| 虚线小盒组 | 多头/GQA 分组示意（查询头 ×H、KV 头 ×K 及共享关系） |

## 维度符号（括号内为默认配置值）

| 符号 | 含义 |
|---|---|
| $B$ / $L$ / $D$ | 批量大小 / 当前序列长度 / 隐藏维度（$D = 768$） |
| $T$ | 词表大小（$T = 6400$, `embed_tokens` 参数行数 / one-hot 维度） |
| $H$ / $K$ / $d$ | 查询头数（$H = 8$）/ KV 头数（$K = 4$, GQA; 若配置为 None 则 $K = H$）/ 头维度（$d = D \div H = 96$） |
| $C_0$ / $C$ | 历史 KV 缓存长度 / 拼接后总长度, $C = C_0 + L$（无缓存时 $C_0 = 0$, $C = L$） |
| $N_{rep}$ | KV 头复制次数, $N_{rep} = H \div K$（默认 2, 即每 2 个查询头共享 1 组 KV） |
| $N$ | Transformer 块层数 = `num_hidden_layers`（默认 8, 图 ① 中 Block 循环 $N$ 次） |
| $I$ | FFN 中间宽度 = `intermediate_size`（默认 $\lceil D\pi/64 \rceil \cdot 64 = 2432$） |
| $E$ | MoE 专家数 = `num_experts`（默认 4）; 每 token 激活 `num_experts_per_tok` = 1 个 |
| $P_{dropout}$ | dropout 概率 = `config.dropout`（默认 0.0, 即默认前向中 dropout 均为恒等） |
| eps | RMSNorm 稳定项 = `rms_norm_eps`（默认 1e-6） |

## 关键结构要点

1. **Embedding 入口**: `input_ids` 以 one-hot 形式 $[B, L, T]$（等价于 int64 编号查表）经 `embed_tokens`（参数 $[T, D]$）得 $[B, L, D]$; 随后接**一次** embedding dropout（模型级, 仅此处）。
2. **Pre-Norm Block ×N**: 每层 `MiniMindBlock` = `input_layernorm → Attention → ⊕残差1 → post_attention_layernorm → FeedForward → ⊕残差2`; 两处残差提供短梯度路径, 子层只学习增量。第 $n+1$ 层的输入是第 $n$ 层 Block 的输出（残差流）。
3. **注意力（Qwen 风格）**: 投影拆头后、RoPE 前, 对 Q/K 做逐头 RMSNorm（V 不做）; RoPE 仅作用于 Q/K; KV 缓存保存**未复制头**的 $[B, C, K, d]$。
4. **多头与 GQA**: $H = 8$ 个查询头并行独立计算 scores; $K = 4$ 组 KV 头, 每组被 $N_{rep} = 2$ 个查询头共享（`repeat_kv` 复制）, 输出 concat 成 $[B, L, H \cdot d]$ 后经 `o_proj` 投回 $D$。
5. **两条等价注意力路径**: 手动路径（缩放点积 → 因果掩码 → padding 遮罩 → softmax → attn_dropout → 加权 V）与 SDPA 融合路径（`flash=True 且 L>1 且无缓存且无 padding` 时）。
6. **FFN = SwiGLU**: $\mathrm{down}(\mathrm{SiLU}(\mathrm{gate}(x)) \cdot \mathrm{up}(x))$, 中间宽度 $I$; `use_moe=True` 时替换为 MoE（gate → softmax → top-1 → 专家 FFN → 加权累加, 训练时附加 `aux_loss` 均衡项）。
7. **输出头**: 最终 RMSNorm 后, `lm_head`（参数 $[T, D]$）把 $D$ 投到词表得 logits $[B, L, T]$; `tie_word_embeddings=True`（默认）时 **lm_head 与 embed_tokens 共享同一权重矩阵**。generate 阶段再经 temperature/top-k/top-p 采样出下一个 token。

## 组件与源码位置交叉索引

源码文件统一为 `model/model_minimind.py`（行号基于当前 master; 教学注释提交后的行号, 若代码变动请按符号名搜索）:

| 图中组件 | 源码位置 | 说明 |
|---|---|---|
| 全部超参数定义 | `MiniMindConfig` L22–84 | $D$/$N$/$H$/$K$/$d$/$T$/$I$/$E$、dropout、RoPE、YaRN、MoE 配置 |
| RMSNorm 模块 | `RMSNorm` L91–107 | `norm()`: $x \cdot \mathrm{rsqrt}(\mathrm{mean}(x^2)+\varepsilon)$; float32 计算后还原 dtype |
| RoPE 频率表 | `precompute_freqs_cis` L113–135 | cos/sin 表 $[\mathrm{max\_pos}, d]$; YaRN 分频插值可选 |
| RoPE 应用 | `apply_rotary_pos_emb` L140–146 | $q \cdot \mathrm{cos} + \mathrm{rotate\_half}(q) \cdot \mathrm{sin}$ |
| GQA 复制 | `repeat_kv` L151–156 | $[B, C, K, d] \to [B, C, H, d]$, expand+reshape |
| q/k/v/o_proj 定义 | `Attention.__init__` L173–176 | 四个无 bias Linear |
| q_norm / k_norm 定义 | `Attention.__init__` L177–178 | 逐头 RMSNorm(d) |
| attn_dropout / resid_dropout | `Attention.__init__` L179–180 | Dropout($P_{dropout}$) |
| flash 标志 | `Attention.__init__` L182 | SDPA 可用性 + `config.flash_attn` |
| 投影 + view 拆头 | `Attention.forward` L191–194 | $[B,L,\cdot] \to [B,L,\cdot,d]$ |
| q_norm/k_norm 应用 | `Attention.forward` L195 |  |
| RoPE 应用（Q/K） | `Attention.forward` L196–197 |  |
| KV 缓存拼接 | `Attention.forward` L198–202 | `torch.cat(dim=1)`, $C = C_0 + L$ |
| past_kv 返回 | `Attention.forward` L203 | `use_cache=True` 时返回未复制头的 K/V |
| transpose + repeat_kv | `Attention.forward` L206 | 对齐为 $[B,H,\cdot,d]$ |
| SDPA 融合路径 | `Attention.forward` L210–211 | 条件: flash 且 $L>1$ 且无缓存且 mask 全 1 |
| 手动注意力路径 | `Attention.forward` L215–222 | 缩放点积 / 因果掩码 / padding 遮罩 / softmax / attn_dropout / @xv |
| 合并头 + o_proj + resid_dropout | `Attention.forward` L224–225 |  |
| 返回 (output, past_kv) | `Attention.forward` L226 |  |
| FeedForward（SwiGLU） | `FeedForward` L232–246 | gate L238 / down L239 / up L240 / SiLU L241 / forward L245–246 |
| MoE 门控 | `MOEFeedForward` L256, L263–269 | gate: $D \to E$, softmax |
| MoE top-k 与直通 | `MOEFeedForward` L270–277 | k=1, `norm_topk_prob` 直通估计 |
| MoE 专家执行 | `MOEFeedForward` L278–288 | 仅被选中 token 过对应专家, `index_add_` 累加 |
| MoE aux_loss | `MOEFeedForward` L289–296 | 负载均衡辅助损失（训练时） |
| Block 组装 | `MiniMindBlock.__init__` L304–309 | attn L306 / input_layernorm L307 / post_attention_layernorm L308 / mlp L309 |
| 残差 1 | `MiniMindBlock.forward` L314, L321 | `residual = h; h += residual`（注意力后） |
| 残差 2 | `MiniMindBlock.forward` L322 | `h = h + mlp(...)`（FFN 后） |
| embed_tokens / 层列表 / 最终 norm | `MiniMindModel.__init__` L334–337 | Embedding L334 / dropout L335 / layers L336 / norm L337 |
| RoPE 缓冲区 | `MiniMindModel.__init__` L338–340 | `freqs_cos/sin` buffer, persistent=False |
| embedding + dropout | `MiniMindModel.forward` L351 | `dropout(embed_tokens(input_ids))`, 仅一次 |
| 位置切片 | `MiniMindModel.forward` L356–359 | 按 `start_pos` 切 cos/sin（含 meta-device 重算） |
| 层循环 ×N | `MiniMindModel.forward` L361–369 | 逐层调 Block, 收集 presents |
| 最终 norm / aux 汇总 | `MiniMindModel.forward` L370–371 |  |
| lm_head | `MiniMindForCausalLM.__init__` L385–386 | Linear $D \to T$; `tie_word_embeddings=True` 时与 embed_tokens 共享权重（L378 声明） |
| logits / 交叉熵 | `MiniMindForCausalLM.forward` L392–405 | logits L396; 训练损失（错位一位, -100 忽略）L398–404 |
| generate 采样 | `MiniMindForCausalLM.generate` L413–455 | temperature / 重复惩罚 / top-k / top-p / 多项式采样 L428–446 |

仓库内其他相关入口: `eval_llm.py`（CLI 推理）、`scripts/convert_model.py`（权重转 Transformers 格式）、`scripts/web_demo.py`（StreamLit 演示）、`trainer/train_pretrain.py` 等（各阶段训练脚本）。

## 数据流速记

1. `input_ids` $[B,L,T]$ → **embed** → dropout（一次）→ $[B,L,D]$
2. **×N 层**: LN → **Attention**（投影→逐头RMSNorm→RoPE→[KV缓存]→GQA→七步核心）→ ⊕ → LN → **FFN(SwiGLU/MoE)** → ⊕
3. 最终 RMSNorm → **lm_head**（与 embed 共享权重）→ logits $[B,L,T]$
4. generate: softmax → probs $[B, T]$ → temperature/top-k/top-p 采样 → next_token $[B, 1]$, 追加到 input_ids 循环生成, 直至 EOS
