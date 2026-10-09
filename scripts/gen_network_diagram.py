#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 MiniMind 完整网络结构图（SVG + 可拖动 HTML）。

用法:
    .venv/bin/python scripts/gen_network_diagram.py                     # SVG + HTML
    .venv/bin/python scripts/gen_network_diagram.py --png /tmp/p.png    # 附 PNG 预览

内容分三节:
  ① 整体结构 MiniMindForCausalLM: embedding → dropout → ×N 层 Block
     （input_layernorm → Attention → ⊕残差 → post_attention_layernorm →
       FeedForward → ⊕残差）→ 最终 RMSNorm → lm_head → logits → softmax/采样
  ② Attention 内部: 投影/RMSNorm/RoPE/KV 缓存/多头 GQA 分组/注意力核心七步
  ③ FeedForward 内部: SwiGLU（gate/up/SiLU/⊗/down）与 MoE 替换（gate/top-k/专家）

约定:
- 平行四边形（浅灰）= 网络结构块; 矩形（白底）= 激活值; 圆 = 逐元素算子
- 实线 = 前向主路径; 虚线 = 条件路径（缓存/可选输入/循环/训练项）
- 公式 mathtext(Computer Modern); 文本转字形路径, 查看器不会重排换行
- 全局内容居中; HTML 版支持拖动节点、连线跟随、KaTeX 渲染
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FAMILY = "sans-serif"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    FAMILY = "Arial Unicode MS"

plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "cm",
    "svg.fonttype": "path",
})

STROKE = "#333333"
FILL_STRUCT = "#e8e8e8"
FILL_TENSOR = "#ffffff"
TXT = "#111111"
ARROW = "#444444"
DASHC = "#777777"
FS = 11.0
PAD_X, PAD_Y = 15.0, 9.0
SKEW = 11.0
GAP_Y = 30.0
COL_GAP = 52.0
SIDE_GAP = 64.0
MARGIN = 36.0
SECTION_GAP = 56.0

# ============================================================ 节点定义
NODES = {
    # ---------- ① 整体结构 ----------
    "IDS": ("rect", [r"input_ids: $[B, L, T]$", "token 的 one-hot 表示",
                     "（等价于编号查表, 编号 $\\in [0, T)$）"]),
    "EMB": ("para", ["embed_tokens:", "Embedding(T, D)", r"参数 $[T, D]$",
                     "one-hot 选择行（等价查表）"]),
    "HID": ("rect", [r"hidden_states: $[B, L, D]$"]),
    "EDROP": ("para", [r"dropout: Dropout($P_{dropout}$)"]),
    "HID2": ("rect", [r"hidden_states: $[B, L, D]$", "（Block 输入 / 残差源）"]),
    "LN1": ("para", ["input_layernorm:", r"RMSNorm(D), 参数 $[D]$"]),
    "XA": ("rect", [r"x: $[B, L, D]$"]),
    "ATTNREF": ("para", ["self_attn: Attention", "（内部展开见 ②）",
                         r"输出 $[B, L, D]$"]),
    "ADD1": ("circ", ["+"]),
    "H1": ("rect", [r"hidden_states: $[B, L, D]$", "（残差相加后）"]),
    "LN2": ("para", ["post_attention_layernorm:", r"RMSNorm(D), 参数 $[D]$"]),
    "XF": ("rect", [r"x: $[B, L, D]$"]),
    "FFNREF": ("para", ["mlp: FeedForward（SwiGLU）", "或 MOEFeedForward",
                        "（内部展开见 ③）", r"输出 $[B, L, D]$"]),
    "ADD2": ("circ", ["+"]),
    "H2": ("rect", [r"hidden_states: $[B, L, D]$", "MiniMindBlock 输出"]),
    "FNORM": ("para", ["norm: RMSNorm(D)", r"参数 $[D]$, 最终归一化"]),
    "LMHEAD": ("para", ["lm_head: Linear", r"参数 $[T, D]$, $D \to T$",
                        "（与 embed_tokens 权重共享）"]),
    "LOGITS": ("rect", [r"logits: $[B, L, T]$", "未归一化分数"]),
    "PROBN": ("para", [r"softmax → probs: $[B, T]$",
                       "temperature / top-p / top-k 采样",
                       "（generate 阶段, 逐 token）"]),
    "NEXTT": ("rect", [r"next_token: $[B, 1]$",
                       "（采样自 probs, 追加到",
                       "input_ids 循环生成, 直至 EOS）"]),

    # ---------- ② Attention 内部 ----------
    "XIN": ("rect", [r"x: $[B, L, D]$",
                     "（= ① 中 input_layernorm 的输出）"]),
    "PE": ("rect", ["position_embeddings", "= (cos, sin)",
                    r"各 $[L, d]$, 由 RoPE 缓冲区", "[max_pos, d]", "按 start_pos 切片"]),
    "PASTK": ("rect", [r"past_k: $[B, C_0, K, d]$", "推理期历史 KV 缓存"]),
    "PASTV": ("rect", [r"past_v: $[B, C_0, K, d]$"]),
    "MASK": ("rect", [r"attention_mask: $[B, C]$", "1=可见, 0=填充", "（可选输入）"]),
    "QP": ("para", ["q_proj: Linear", r"参数 $[H \cdot d, D]$", r"$D \to H \cdot d$",
                    "view 拆头"]),
    "KP": ("para", ["k_proj: Linear", r"参数 $[K \cdot d, D]$", r"$D \to K \cdot d$",
                    "view 拆头"]),
    "VP": ("para", ["v_proj: Linear", r"参数 $[K \cdot d, D]$", r"$D \to K \cdot d$",
                    "view 拆头"]),
    "XQ": ("rect", [r"xq: $[B, L, H, d]$"]),
    "XK": ("rect", [r"xk: $[B, L, K, d]$"]),
    "XV": ("rect", [r"xv: $[B, L, K, d]$", "不做 RMSNorm、不做 RoPE"]),
    "QN": ("para", ["q_norm: RMSNorm(d)", r"参数 $[d]$", "逐头归一化最后一维",
                    r"$w \cdot x \cdot \mathrm{rsqrt}(\mathrm{mean}(x^2)+\mathrm{eps})$",
                    "float32 计算后还原 dtype"]),
    "KN": ("para", ["k_norm: RMSNorm(d)", r"参数 $[d]$", "逐头归一化最后一维"]),
    "XQ2": ("rect", [r"xq: $[B, L, H, d]$"]),
    "XK2": ("rect", [r"xk: $[B, L, K, d]$"]),
    "ROPE": ("para", ["RoPE: apply_rotary_pos_emb", "（仅 Q/K）",
                      r"$q \cdot \mathrm{cos} +$ rotate_half(q) $\cdot \mathrm{sin}$",
                      "（K 同理）", r"cos/sin: $[L, d] \to [L, 1, d]$", "广播到各头"]),
    "XQ3": ("rect", [r"xq: $[B, L, H, d]$"]),
    "XK3": ("rect", [r"xk: $[B, L, K, d]$"]),
    "KVCAT": ("para", ["KV 缓存拼接", "（仅推理且 past_key_value 非空;",
                       r"无缓存时直通 $C = L$）",
                       "xk = torch.cat([past_k, xk], dim=1)",
                       "xv = torch.cat([past_v, xv], dim=1)"]),
    "XK4": ("rect", [r"xk: $[B, C, K, d]$", r"$C = C_0 + L$"]),
    "XV2": ("rect", [r"xv: $[B, C, K, d]$"]),
    "PASTKV": ("rect", ["past_kv = (xk, xv)", r"$[B, C, K, d]$",
                        "use_cache=True 时返回", "（未复制头的 K/V）"]),
    "GQAP": ("gqa", []),
    "ATTN": ("para", [
        "Attention 核心计算",
        "① repeat_kv: 复制 K/V 头",
        r"$N_{rep} = H \div K$ 次,",
        r"对齐为 $[B, H, L, d]$",
        r"与 $[B, H, C, d]$",
        r"② scores $= xq \cdot xk^{\top} \div \sqrt{d}$（对 H 个头并行独立）",
        r"$\to [B, H, L, C]$（float32）",
        r"③ + 因果掩码: 最后 $L$ 个键列",
        r"加 $[L, L]$ 上三角 $-\infty$",
        "④ + padding 遮罩:",
        r"$(1-\mathrm{mask}) \cdot (-10^{9})$",
        r"$[B, C] \to [B, 1, 1, C]$ 广播",
        "⑤ softmax(dim=-1),",
        r"再 attn_dropout($P_{dropout}$)",
        r"⑥ attn @ xv $\to [B, H, L, d]$",
        "⑦ 合并 H 个头: transpose(1,2)",
        r"+ reshape $\to [B, L, H \cdot d]$",
        "（②–⑥ 满足条件时由",
        "F.scaled_dot_product_attention",
        "is_causal=True 一步融合:",
        r"flash 且 $L > 1$ 且无缓存",
        "且无 padding）",
    ]),
    "OUTH": ("rect", [r"output: $[B, L, H \cdot d]$"]),
    "OP": ("para", ["o_proj: Linear", r"参数 $[D, H \cdot d]$", r"$H \cdot d \to D$"]),
    "RDR": ("para", ["resid_dropout:", r"Dropout($P_{dropout}$)"]),
    "OUT": ("rect", [r"output: $[B, L, D]$", "（回到 ① 的 ⊕ 残差相加）"]),
    "RET": ("rect", ["返回 (output, past_kv)"]),

    # ---------- ③ FeedForward / MoE ----------
    "FFIN": ("rect", [r"x: $[B, L, D]$", "（post_attention_layernorm 输出）"]),
    "GP": ("para", ["gate_proj: Linear", r"参数 $[I, D]$", r"$D \to I$"]),
    "UP": ("para", ["up_proj: Linear", r"参数 $[I, D]$", r"$D \to I$"]),
    "GACT": ("para", ["SiLU(gate)", r"$z \cdot \mathrm{sigmoid}(z)$"]),
    "MUL": ("circ", ["×"]),
    "HMID": ("rect", [r"h: $[B, L, I]$", r"$I$ = intermediate_size",
                      r"（默认 $\lceil D\pi/64 \rceil \cdot 64 = 2432$）"]),
    "DP": ("para", ["down_proj: Linear", r"参数 $[D, I]$", r"$I \to D$"]),
    "FFOUT": ("rect", [r"output: $[B, L, D]$"]),
    "MOEX": ("rect", [r"x_flat: $[B \cdot L, D]$"]),
    "MOEG": ("para", ["gate: Linear", r"参数 $[E, D]$, $D \to E$"]),
    "MOESC": ("rect", [r"scores: $[B \cdot L, E]$", "softmax(dim=-1)"]),
    "MOETK": ("para", ["top-k（k=1）+ 权重归一化", "norm_topk_prob 直通估计"]),
    "MOEEX": ("para", ["专家 $e_1 \\dots e_E$（E=4）", "每个 = FeedForward(SwiGLU)",
                       "仅执行被选中 token"]),
    "MOESUM": ("circ", ["Σ"]),
    "MOEOUT": ("rect", [r"y: $[B, L, D]$"]),
    "MOEAUX": ("rect", ["aux_loss: 负载均衡辅助项", "（训练时加权加入总损失）"]),
}

# 三列分支中各节点所在列: 0=左 1=中 2=右（仅 ② ③ 使用）
COLMAP = {"QP": 0, "KP": 1, "VP": 2, "XQ": 0, "XK": 1, "XV": 2,
          "QN": 0, "KN": 1, "XQ2": 0, "XK2": 1, "XQ3": 0, "XK3": 1,
          "XK4": 1, "XV2": 2,
          "GP": 0, "UP": 2, "GACT": 0, "MUL": 1}

SECTIONS = [
    {"title": "① 整体结构（MiniMindForCausalLM 前向）", "rows": [
        ["IDS"], ["EMB"], ["HID"], ["EDROP"], ["HID2"],
        ["LN1"], ["XA"], ["ATTNREF"], ["ADD1"], ["H1"],
        ["LN2"], ["XF"], ["FFNREF"], ["ADD2"], ["H2"],
        ["FNORM"], ["LMHEAD"], ["LOGITS"], ["PROBN"], ["NEXTT"],
     ], "sides": {}},
    {"title": "② Attention 内部结构（self_attn 展开, 多头分组见右侧）", "rows": [
        ["XIN"],
        ["QP", "KP", "VP"], ["XQ", "XK", "XV"], ["QN", "KN"], ["XQ2", "XK2"],
        ["ROPE"], ["XQ3", "XK3"], ["KVCAT"], ["XK4", "XV2"], ["ATTN"],
        ["OUTH"], ["OP"], ["RDR"], ["OUT"], ["RET"],
     ], "sides": {
        "PE": ("left", "ROPE", 0),
        "PASTK": ("left", "KVCAT", -42), "PASTV": ("left", "KVCAT", 42),
        "MASK": ("right", "ATTN", 0), "PASTKV": ("right", "XV2", 0),
        "GQAP": ("right", "XV", 0),
     }},
    {"title": "③ FeedForward 内部（SwiGLU, 默认）", "rows": [
        ["FFIN"], ["GP", "UP"], ["GACT", "MUL"], ["HMID"], ["DP"], ["FFOUT"],
     ], "sides": {}},
    {"title": "④ MoE 替换（use_moe=True 时替换上面的 mlp）", "rows": [
        ["MOEX"], ["MOEG"], ["MOESC"], ["MOETK"], ["MOEEX"], ["MOESUM"],
        ["MOEOUT"], ["MOEAUX"],
     ], "sides": {}},
]

BLOCK_IDS = ["LN1", "XA", "ATTNREF", "ADD1", "H1", "LN2", "XF", "FFNREF", "ADD2", "H2"]

# 边: (src, dst, dashed, src_anchor, dst_anchor, label, kind)
# kind: None=自动正交（同行横向直连 / 跨行走行间通道） / topch=上方通道 /
#       bypassL:x / bypassR:x=侧向旁路（残差、层循环、跨节引用）
EDGES = [
    # ① 主链
    ("IDS", "EMB", 0, None, None, None, None),
    ("EMB", "HID", 0, None, None, None, None),
    ("HID", "EDROP", 0, None, None, None, None),
    ("EDROP", "HID2", 0, None, None, None, None),
    ("HID2", "LN1", 0, None, None, None, None),
    ("LN1", "XA", 0, None, None, None, None),
    ("XA", "ATTNREF", 0, None, None, None, None),
    ("ATTNREF", "ADD1", 0, None, None, None, None),
    ("ADD1", "H1", 0, None, None, None, None),
    ("H1", "LN2", 0, None, None, None, None),
    ("LN2", "XF", 0, None, None, None, None),
    ("XF", "FFNREF", 0, None, None, None, None),
    ("FFNREF", "ADD2", 0, None, None, None, None),
    ("ADD2", "H2", 0, None, None, None, None),
    ("H2", "FNORM", 0, None, None, None, None),
    ("FNORM", "LMHEAD", 0, None, None, None, None),
    ("LMHEAD", "LOGITS", 0, None, None, None, None),
    ("LOGITS", "PROBN", 1, None, None, None, None),
    ("PROBN", "NEXTT", 1, None, None, None, None),
    # ② 输入张量与投影
    ("XIN", "QP", 0, None, None, None, None),
    ("XIN", "KP", 0, None, None, None, None),
    ("XIN", "VP", 0, None, None, None, None),
    # ① 残差跳连与层循环（正交旁路折线, 不穿组件）
    ("HID2", "ADD1", 0, ("left", 0.5), ("left", 0.5), "残差 1", "bypassL:60"),
    ("H1", "ADD2", 0, ("right", 0.5), ("right", 0.5), "残差 2", "bypassR:60"),
    ("H2", "HID2", 1, ("left", 0.5), ("left", 0.5), "×N=8 层循环", "bypassL:150"),
    # ② 投影与归一化
    ("QP", "XQ", 0, None, None, None, None),
    ("KP", "XK", 0, None, None, None, None),
    ("VP", "XV", 0, None, None, None, None),
    ("XQ", "QN", 0, None, None, None, None),
    ("XK", "KN", 0, None, None, None, None),
    ("QN", "XQ2", 0, None, None, None, None),
    ("KN", "XK2", 0, None, None, None, None),
    ("XQ2", "ROPE", 0, ("bottom", 0.5), ("top", 0.28), None, None),
    ("XK2", "ROPE", 0, ("bottom", 0.5), ("top", 0.72), None, None),
    ("PE", "ROPE", 0, ("right", 0.5), ("left", 0.5), None, None),
    ("ROPE", "XQ3", 0, ("bottom", 0.28), ("top", 0.5), None, None),
    ("ROPE", "XK3", 0, ("bottom", 0.72), ("top", 0.5), None, None),
    ("XK3", "KVCAT", 0, None, None, None, None),
    ("KVCAT", "XK4", 0, ("bottom", 0.5), ("top", 0.5), None, None),
    ("KVCAT", "XV2", 0, ("bottom", 0.8), ("top", 0.5), None, None),
    ("XK4", "ATTN", 0, None, None, None, None),
    ("ATTN", "OUTH", 0, None, None, None, None),
    ("OUTH", "OP", 0, None, None, None, None),
    ("OP", "RDR", 0, None, None, None, None),
    ("RDR", "OUT", 0, None, None, None, None),
    ("OUT", "RET", 0, None, None, None, None),
    # ② 条件路径
    ("PASTK", "KVCAT", 1, ("right", 0.5), ("left", 0.3), None, None),
    ("PASTV", "KVCAT", 1, ("right", 0.5), ("left", 0.7), None, None),
    ("MASK", "ATTN", 1, ("left", 0.5), ("right", 0.5), "非空时", None),
    ("XK4", "PASTKV", 1, ("top", 0.5), ("top", 0.25), None, "topch"),
    ("XV2", "PASTKV", 1, ("right", 0.7), ("left", 0.7), None, None),
    ("XQ", "GQAP", 1, ("top", 0.5), ("top", 0.2), None, "topch"),
    ("XK", "GQAP", 1, ("top", 0.5), ("top", 0.5), None, "topch"),
    ("XV", "GQAP", 1, ("right", 0.5), ("left", 0.5), None, None),
    # ② 绕行折线（dropin: 下行 → 沿目标上方通道 → 垂直入顶）
    ("XV", "KVCAT", 0, None, None, None, "dropin:0.75"),
    ("XQ3", "ATTN", 0, None, None, None, "dropin:0.2"),
    ("XV2", "ATTN", 0, None, None, None, "dropin:0.8"),
    ("PASTKV", "RET", 1, ("bottom", 0.5), ("right", 0.5), None, "bypassR:30"),
    # ③ SwiGLU
    ("FFIN", "GP", 0, ("bottom", 0.25), ("top", 0.5), None, None),
    ("FFIN", "UP", 0, ("bottom", 0.75), ("top", 0.5), None, None),
    ("GP", "GACT", 0, None, None, None, None),
    ("GACT", "MUL", 0, ("right", 0.5), ("left", 0.5), None, None),
    ("UP", "MUL", 0, ("bottom", 0.5), ("top", 0.5), None, None),
    ("MUL", "HMID", 0, None, None, None, None),
    ("HMID", "DP", 0, None, None, None, None),
    ("DP", "FFOUT", 0, None, None, None, None),
    # ④ MoE（虚线 = 替换路径; 右侧旁路避开 ③ 的 gate/up 分支）
    ("FFIN", "MOEX", 1, ("right", 0.5), ("right", 0.5), "use_moe=True 替换", "bypassR:250"),
    ("MOEX", "MOEG", 1, None, None, None, None),
    ("MOEG", "MOESC", 1, None, None, None, None),
    ("MOESC", "MOETK", 1, None, None, None, None),
    ("MOETK", "MOEEX", 1, None, None, None, None),
    ("MOEEX", "MOESUM", 1, None, None, None, None),
    ("MOESUM", "MOEOUT", 1, None, None, None, None),
    ("MOESUM", "MOEAUX", 1, ("right", 0.5), ("left", 0.5), None, "bypassR:40"),
]

TITLE = "MiniMind 完整网络结构图（Embedding → ×N Block → lm_head）"


class Box:
    def __init__(self, nid, shape, lines):
        self.id, self.shape, self.lines = nid, shape, lines
        self.x = self.y = self.w = self.h = self.line_h = 0.0

    @property
    def top(self):
        return self.y - self.h / 2

    @property
    def bottom(self):
        return self.y + self.h / 2

    @property
    def left(self):
        return self.x - self.w / 2

    @property
    def right(self):
        return self.x + self.w / 2

    def anchor(self, side, frac=0.5):
        if side == "top":
            return (self.left + frac * self.w, self.top)
        if side == "bottom":
            return (self.left + frac * self.w, self.bottom)
        if side == "left":
            return (self.left, self.top + frac * self.h)
        return (self.right, self.top + frac * self.h)


def measure(fig, ax):
    renderer = fig.canvas.get_renderer()
    boxes = {}
    for nid, (shape, lines) in NODES.items():
        if shape == "gqa":
            b = Box(nid, shape, [])
            b.w, b.h = 400.0, 196.0
            boxes[nid] = b
            continue
        if shape == "circ":
            b = Box(nid, shape, lines)
            b.w = b.h = 24.0
            boxes[nid] = b
            continue
        ws, hmax = [], 0.0
        for s in lines:
            t = ax.text(0, 0, s, fontsize=FS)
            bb = t.get_window_extent(renderer=renderer)
            ws.append(bb.width)
            hmax = max(hmax, bb.height)
            t.remove()
        b = Box(nid, shape, lines)
        b.line_h = hmax + 6.0
        b.w = max(ws) + 2 * PAD_X + (2 * SKEW if shape == "para" else 0)
        b.h = len(lines) * b.line_h + 2 * PAD_Y
        boxes[nid] = b
    return boxes


def layout(boxes):
    """逐节布局: 行下移 + 侧列定位; 最后全局水平居中。返回 (W, H, 节标题位置)。"""
    y = MARGIN + 40.0
    titles = []
    for sec in SECTIONS:
        titles.append((sec["title"], y))
        y += 22.0
        col_ids = [n for row in sec["rows"] for n in row if n in COLMAP]
        center_ids = [n for row in sec["rows"] for n in row if n not in COLMAP]
        colw = max((boxes[n].w for n in col_ids), default=0.0)
        centerw = max((boxes[n].w for n in center_ids), default=0.0)
        branch_extent = (colw + COL_GAP + colw / 2) if colw else 0.0
        halfw = max(branch_extent, centerw / 2)
        cx = [-(colw + COL_GAP), 0.0, colw + COL_GAP] if colw else [0.0] * 3
        side_by_ref = {}
        for nid, (side, ref, dy) in sec["sides"].items():
            side_by_ref.setdefault(ref, []).append(boxes[nid].h + 2 * abs(dy))
        for row in sec["rows"]:
            hrow = max(boxes[n].h for n in row)
            for nid in row:
                hrow = max([hrow] + side_by_ref.get(nid, []))
            for nid in row:
                b = boxes[nid]
                b.y = y + hrow / 2
                b.x = cx[COLMAP[nid]] if nid in COLMAP else 0.0
            y += hrow + GAP_Y
        for nid, (side, ref, dy) in sec["sides"].items():
            b, r = boxes[nid], boxes[ref]
            b.x = -(halfw + SIDE_GAP + b.w / 2) if side == "left" \
                else (halfw + SIDE_GAP + b.w / 2)
            b.y = r.y + dy
        y += SECTION_GAP - GAP_Y
    H = y + 70.0
    lo = min(b.left for b in boxes.values())
    hi = max(b.right for b in boxes.values())
    W = hi - lo + 2 * MARGIN
    dx = (W - (hi - lo)) / 2 - lo
    for b in boxes.values():
        b.x += dx
    return W, H, titles


def elbow_pts(name, B):
    raise NotImplementedError  # 已由 route() 的 dropin/bypass 通吃, 保留占位


def route(b0, b1, sa, ta, kind):
    """正交折线路由: 同行横向直连; 跨行走行间通道; 旁路走组件外侧。不穿过组件。"""
    p0 = b0.anchor(*sa)
    p1 = b1.anchor(*ta)
    if kind is None:
        s, t = sa[0], ta[0]
        if s == "bottom" and t == "top":
            if abs(p0[0] - p1[0]) < 1:
                return [p0, p1]
            ym = (p0[1] + p1[1]) / 2
            return [p0, (p0[0], ym), (p1[0], ym), p1]
        if s in ("left", "right") and t in ("left", "right"):
            if abs(p0[1] - p1[1]) < 1:
                return [p0, p1]
            xm = (p0[0] + p1[0]) / 2
            return [p0, (xm, p0[1]), (xm, p1[1]), p1]
        return [p0, p1]
    if kind == "topch":
        ych = min(p0[1], p1[1]) - 18
        return [p0, (p0[0], ych), (p1[0], ych), p1]
    if kind.startswith("dropin"):
        fx = float(kind.split(":")[1])
        ym = b1.top - 16
        ex = b1.left + fx * b1.w
        return [p0, (p0[0], ym), (ex, ym), (ex, b1.top)]
    if kind.startswith("bypassL") or kind.startswith("bypassR"):
        off = float(kind.split(":")[1])
        if kind.startswith("bypassL"):
            bx = min(b0.left, b1.left) - off
        else:
            bx = max(b0.right, b1.right) + off
        return [p0, (bx, p0[1]), (bx, p1[1]), p1]
    return [p0, p1]


def draw_arrow(ax, p0, p1, dashed, zorder=2):
    color = DASHC if dashed else ARROW
    props = dict(arrowstyle="-|>", color=color, lw=1.2,
                 shrinkA=0, shrinkB=0, mutation_scale=13)
    if dashed:
        props["linestyle"] = (0, (4, 3))
    ax.annotate("", xy=p1, xytext=p0, arrowprops=props, zorder=zorder)


def draw_gqa(ax, b):
    """多头/GQA 示意: 上排 H=8 查询头, 下排 K=4 KV 头, 每个 KV 头连 2 个查询头。"""
    ax.add_patch(Rectangle((b.left, b.top), b.w, b.h, facecolor="#ffffff",
                           edgecolor=STROKE, lw=1.0, linestyle=(0, (4, 3))))
    ax.text(b.x, b.top + 14, "多头拆分与 GQA 分组（H=8, K=4）", fontsize=10.5,
            color=TXT, ha="center", va="center", fontweight="bold")
    gw, gh, gp = 38.0, 17.0, 7.0
    row1_y = b.top + 46
    row2_y = b.top + 98
    total = 8 * gw + 7 * gp
    x0 = b.x - total / 2
    qx = []
    for i in range(8):
        gx = x0 + i * (gw + gp)
        qx.append(gx + gw / 2)
        ax.add_patch(Rectangle((gx, row1_y), gw, gh, facecolor=FILL_STRUCT,
                               edgecolor=STROKE, lw=0.8))
        ax.text(gx + gw / 2, row1_y + gh / 2, f"$h_{{{i + 1}}}$", fontsize=8,
                ha="center", va="center")
    ktot = 4 * gw + 3 * gp
    k0 = b.x - ktot / 2
    kx = []
    for i in range(4):
        gx = k0 + i * (gw + gp) + (gw + gp) / 2
        kx.append(gx + gw / 2)
        ax.add_patch(Rectangle((gx, row2_y), gw, gh, facecolor="#ffffff",
                               edgecolor=STROKE, lw=0.8))
        ax.text(gx + gw / 2, row2_y + gh / 2, f"$kv_{{{i + 1}}}$", fontsize=8,
                ha="center", va="center")
    for i in range(4):
        for j in (2 * i, 2 * i + 1):
            ax.annotate("", xy=(qx[j], row1_y + gh + 1), xytext=(kx[i], row2_y - 1),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(b.x - total / 2, row1_y - 10, "查询头 ×H=8（每头 d 维）", fontsize=9,
            ha="left", va="bottom")
    ax.text(b.x - total / 2, row2_y - 10, "KV 头 ×K=4", fontsize=9,
            ha="left", va="bottom")
    ax.text(b.x, b.bottom - 26,
            r"$N_{rep} = H \div K = 2$: 每 2 个查询头共享 1 组 KV",
            fontsize=9.5, ha="center", va="center")
    ax.text(b.x, b.bottom - 10,
            "q_norm/k_norm/RoPE 逐头独立; scores 对 H 个头并行独立, 输出 concat",
            fontsize=9, ha="center", va="center")


def draw(fig, ax, boxes, W, H, titles):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    bl = [boxes[n] for n in BLOCK_IDS]
    x0 = min(b.left for b in bl) - 90
    x1 = max(b.right for b in bl) + 90
    y0 = min(b.top for b in bl) - 26
    y1 = max(b.bottom for b in bl) + 26
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor="#999999", lw=1.1, linestyle=(0, (5, 4)), zorder=0))
    ax.text(x1 - 8, y0 + 14, "MiniMindBlock × N（num_hidden_layers = 8）",
            fontsize=10, color="#555555", ha="right", va="center")

    # ---- 连线层（zorder=2, 在组件之下; 正交折线, 不穿组件） ----
    for (s, t, dashed, sa, ta, label, kind) in EDGES:
        pts = route(boxes[s], boxes[t], sa or ("bottom", 0.5),
                    ta or ("top", 0.5), kind)
        color = DASHC if dashed else ARROW
        ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts], color=color,
                           lw=1.2, zorder=2,
                           linestyle=(0, (4, 3)) if dashed else "solid"))
        draw_arrow(ax, pts[-2], pts[-1], dashed, zorder=2)
        if label:
            if kind and str(kind).startswith("bypass"):
                lx = pts[1][0]
                ly = (pts[0][1] + pts[-1][1]) / 2
                ha = "right" if str(kind).startswith("bypassL") else "left"
                ax.text(lx + (-5 if ha == "right" else 5), ly, label, fontsize=9,
                        color=TXT, ha=ha, va="center", zorder=4)
            else:
                mid = pts[len(pts) // 2]
                ax.text(mid[0] + 7, mid[1] - 8, label, fontsize=9, color=TXT,
                        ha="left", va="bottom", zorder=4)

    # ---- 组件层（zorder>=3, 覆盖在其下穿行的连线） ----
    for b in boxes.values():
        if b.shape == "para":
            pts = [(b.left + SKEW, b.top), (b.right, b.top),
                   (b.right - SKEW, b.bottom), (b.left, b.bottom)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=FILL_STRUCT,
                                 edgecolor=STROKE, lw=1.1, zorder=3))
        elif b.shape == "circ":
            ax.add_patch(Circle((b.x, b.y), b.w / 2, facecolor=FILL_TENSOR,
                                edgecolor=STROKE, lw=1.2, zorder=3))
            ax.text(b.x, b.y, b.lines[0], fontsize=13, color=TXT,
                    ha="center", va="center", zorder=4)
        elif b.shape == "gqa":
            draw_gqa(ax, b)
            continue
        else:
            ax.add_patch(Rectangle((b.left, b.top), b.w, b.h,
                                   facecolor=FILL_TENSOR, edgecolor=STROKE,
                                   lw=1.1, zorder=3))
        if b.shape != "circ":
            for i, s in enumerate(b.lines):
                ty = b.top + PAD_Y + b.line_h * (i + 0.5)
                ax.text(b.x, ty, s, fontsize=FS, color=TXT, ha="center",
                        va="center", zorder=4)

    ax.text(W / 2, MARGIN - 14, TITLE, fontsize=15, color="#000000",
            ha="center", va="center", fontweight="bold")
    for t, ty in titles:
        ax.text(MARGIN, ty, t, fontsize=12.5, color="#333333",
                ha="left", va="center", fontweight="bold")

    ly = H - 34
    lx = MARGIN
    renderer = fig.canvas.get_renderer()
    ax.add_patch(Polygon([(lx + 7, ly - 9), (lx + 46, ly - 9),
                          (lx + 39, ly + 9), (lx, ly + 9)], closed=True,
                         facecolor=FILL_STRUCT, edgecolor=STROKE, lw=1.0))
    t1 = ax.text(lx + 54, ly, "网络结构块（模块 / 参数 / 算子）", fontsize=9.5, va="center")
    lx += 54 + t1.get_window_extent(renderer).width / fig.dpi + 28
    ax.add_patch(Rectangle((lx, ly - 9), 46, 18, facecolor=FILL_TENSOR,
                           edgecolor=STROKE, lw=1.0))
    t2 = ax.text(lx + 54, ly, "激活值（张量）", fontsize=9.5, va="center")
    lx += 54 + t2.get_window_extent(renderer).width / fig.dpi + 28
    ax.add_patch(Circle((lx + 12, ly), 11, facecolor=FILL_TENSOR,
                        edgecolor=STROKE, lw=1.0))
    ax.text(lx + 12, ly, "+", fontsize=11, ha="center", va="center")
    t3 = ax.text(lx + 30, ly, "逐元素算子（⊕残差 / ×门控 / Σ累加）", fontsize=9.5, va="center")
    lx += 30 + t3.get_window_extent(renderer).width / fig.dpi + 28
    draw_arrow(ax, (lx, ly), (lx + 46, ly), 0)
    t4 = ax.text(lx + 54, ly, "前向数据流", fontsize=9.5, va="center")
    lx += 54 + t4.get_window_extent(renderer).width / fig.dpi + 28
    draw_arrow(ax, (lx, ly), (lx + 46, ly), 1)
    ax.text(lx + 54, ly, "条件路径（缓存 / 可选输入 / 层循环 / MoE 替换）", fontsize=9.5,
            va="center")


# ============================================================ HTML（可拖动版）
HTML_TMPL = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>MiniMind 网络结构图（可拖动）</title>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<style>
 body{margin:0;background:#fafafa;font-family:'PingFang SC','Microsoft YaHei',sans-serif}
 #bar{position:fixed;top:0;left:0;right:0;height:40px;background:#fff;border-bottom:1px solid #ddd;
      display:flex;align-items:center;gap:16px;padding:0 16px;font-size:13px;z-index:9}
 #bar button{padding:4px 12px;cursor:pointer}
 #canvas{position:relative;margin:52px auto 30px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.12)}
 #edges{position:absolute;left:0;top:0;pointer-events:none}
 .node{position:absolute;padding:8px 14px;border:1.2px solid #333;font-size:12.5px;
       line-height:1.55;text-align:center;cursor:move;user-select:none;white-space:nowrap;
       transform:translate(-50%,-50%);background:#fff}
 .node.para{background:transparent;border:none;padding:0}
 .node.para>.sk{display:inline-block;transform:skewX(-14deg);border:1.2px solid #333;padding:8px 14px;background:#e8e8e8}
 .node.para>.sk>.inner{display:inline-block;transform:skewX(14deg)}
 .node.circ{border-radius:50%;width:26px;height:26px;padding:0;line-height:24px;
            font-size:14px;font-weight:bold}
 .sec{position:absolute;font-size:14px;color:#333;margin:0;white-space:nowrap;font-weight:bold}
 .hnd{fill:#666;opacity:.3;cursor:move;pointer-events:auto}
 .hnd:hover{opacity:1}
</style>
</head>
<body>
<div id="bar">
  <b>MiniMind 网络结构图</b><span>拖动模块调整布局; 拖动连线中点小圆点手动调整走线（双击圆点恢复自动走线）</span>
  <button onclick="resetLayout()">重置布局</button>
  <span style="color:#888">公式由 KaTeX 渲染（需联网）</span>
</div>
<div id="canvas"><svg id="edges"></svg></div>
<script>
const DATA = %%DATA%%;
const cv = document.getElementById('canvas'), eg = document.getElementById('edges');
const nodes = {};
const GAP=34, CGAP=44, SGAP=56, MARG=40;
function buildDOM(){
  for (const [id, n] of Object.entries(DATA.nodes)) {
    const d = document.createElement('div');
    d.className = 'node ' + (n.shape==='para'?'para':n.shape==='circ'?'circ':'');
    if (n.shape==='para') {
      d.innerHTML = '<span class="sk"><span class="inner">'+
        n.lines.map(l=>'<div>'+l+'</div>').join('')+'</span></span>';
    } else {
      d.innerHTML = n.lines.map(l=>'<div>'+l+'</div>').join('');
    }
    d.dataset.id = id;
    cv.appendChild(d); nodes[id]={el:d, x:0, y:0};
    d.addEventListener('pointerdown', startDrag);
  }
  for (const s of DATA.sections) {
    const h = document.createElement('h2'); h.className='sec'; h.textContent=s.title;
    cv.appendChild(h); s.el=h;
  }
}
function rect(id){ return nodes[id].el.getBoundingClientRect(); }
function own(id){ const r=rect(id), c=cv.getBoundingClientRect();
  return {w:c.width? r.width : r.width, h:r.height}; }
function sizeOf(id){ const r=rect(id); return {w:r.width, h:r.height}; }
function resetLayout(){
  for (const k in HANDLES) delete HANDLES[k];
  let y = 70;
  for (const s of DATA.sections) {
    s.el.style.left = MARG+'px'; s.el.style.top = y+'px'; y += 30;
    let colw = 0, centerW = 0;
    for (const row of s.rows) for (const n of row) {
      if (n in DATA.colmap) colw = Math.max(colw, sizeOf(n).w);
      else centerW = Math.max(centerW, sizeOf(n).w);
    }
    const half = Math.max(colw ? colw + CGAP + colw/2 : 0, centerW/2);
    const sideByRef = {};
    for (const [n, [side, ref, dy]] of Object.entries(s.sides||{}))
      (sideByRef[ref] = sideByRef[ref] || []).push(sizeOf(n).h + 2*Math.abs(dy));
    const cx = c => c===undefined?0:[-(colw+CGAP),0,(colw+CGAP)][c];
    for (const row of s.rows) {
      let hrow = 0;
      for (const n of row) {
        hrow = Math.max(hrow, sizeOf(n).h);
        for (const h of (sideByRef[n]||[])) hrow = Math.max(hrow, h);
      }
      for (const n of row) { nodes[n].y = y + hrow/2; nodes[n].x = cx(DATA.colmap[n]); }
      y += hrow + GAP;
    }
    for (const [n, [side, ref, dy]] of Object.entries(s.sides||{})) {
      nodes[n].y = nodes[ref].y + dy;
      nodes[n].x = side==='left' ? -(half+SGAP+sizeOf(n).w/2) : (half+SGAP+sizeOf(n).w/2);
    }
    y += 30;
  }
  let lo=1e9, hi=-1e9;
  for (const id in nodes){ const s=sizeOf(id); lo=Math.min(lo,nodes[id].x-s.w/2); hi=Math.max(hi,nodes[id].x+s.w/2); }
  const W = hi-lo+2*MARG, dx=(W-(hi-lo))/2-lo;
  let H=0; for (const id in nodes){ nodes[id].x+=dx; H=Math.max(H, nodes[id].y+sizeOf(id).h/2); }
  cv.style.width = W+'px'; cv.style.height=(H+70)+'px';
  placeAll(); drawEdges();
}
function place(id){ const n=nodes[id]; n.el.style.left=n.x+'px'; n.el.style.top=n.y+'px'; }
function placeAll(){ for (const id in nodes) place(id); }
function anchor(id, side, frac){
  const s = sizeOf(id), n = nodes[id];
  frac = frac===undefined?0.5:frac;
  if(side==='top')    return [n.x-s.w/2+frac*s.w, n.y-s.h/2];
  if(side==='bottom') return [n.x-s.w/2+frac*s.w, n.y+s.h/2];
  if(side==='left')   return [n.x-s.w/2, n.y-s.h/2+frac*s.h];
  return [n.x+s.w/2, n.y-s.h/2+frac*s.h];
}
function routePts(e){
  const sa=e.sa||['bottom',.5], ta=e.ta||['top',.5];
  const p0=anchor(e.s,sa[0],sa[1]), p1=anchor(e.t,ta[0],ta[1]);
  const kind=e.kind;
  if(!kind){
    if(sa[0]=='bottom'&&ta[0]=='top'){
      if(Math.abs(p0[0]-p1[0])<1) return [p0,p1];
      const ym=(p0[1]+p1[1])/2; return [p0,[p0[0],ym],[p1[0],ym],p1];
    }
    if(['left','right'].includes(sa[0])&&['left','right'].includes(ta[0])){
      if(Math.abs(p0[1]-p1[1])<1) return [p0,p1];
      const xm=(p0[0]+p1[0])/2; return [p0,[xm,p0[1]],[xm,p1[1]],p1];
    }
    return [p0,p1];
  }
  if(kind=='topch'){
    const ych=Math.min(p0[1],p1[1])-14;
    return [p0,[p0[0],ych],[p1[0],ych],p1];
  }
  if(kind.startsWith('dropin')){
    const fx=parseFloat(kind.split(':')[1]);
    const st=sizeOf(e.t), nt=nodes[e.t];
    const ym=nt.y-st.h/2-14, ex=nt.x-st.w/2+fx*st.w;
    return [p0,[p0[0],ym],[ex,ym],[ex,nt.y-st.h/2]];
  }
  if(kind.startsWith('bypass')){
    const off=parseFloat(kind.split(':')[1]);
    const s0=sizeOf(e.s), s1=sizeOf(e.t), n0=nodes[e.s], n1=nodes[e.t];
    const bx=kind.startsWith('bypassL')
      ? Math.min(n0.x-s0.w/2, n1.x-s1.w/2)-off
      : Math.max(n0.x+s0.w/2, n1.x+s1.w/2)+off;
    return [p0,[bx,p0[1]],[bx,p1[1]],p1];
  }
  return [p0,p1];
}
const HANDLES = {};  // edgeIdx -> {x, y}: 手动调整的连线通道点
function edgePath(e, idx){
  const sa=e.sa||['bottom',.5], ta=e.ta||['top',.5];
  const p0=anchor(e.s,sa[0],sa[1]), p1=anchor(e.t,ta[0],ta[1]);
  const h = HANDLES[idx];
  let pts;
  if (h) {  // 手动模式: 保持正交, 通道经过手柄所在的 x（垂直类）或 y（水平类）
    if (['left','right'].includes(sa[0])) pts=[p0,[p0[0],h.y],[p1[0],h.y],p1];
    else pts=[p0,[h.x,p0[1]],[h.x,p1[1]],p1];
  } else pts=routePts(e);
  return {d:'M '+pts.map(p=>p[0]+' '+p[1]).join(' L '), pts};
}
function drawEdges(){
  eg.setAttribute('width', cv.clientWidth); eg.setAttribute('height', cv.clientHeight);
  eg.innerHTML='<defs></defs>';
  const defs=eg.querySelector('defs');
  for (const [id,c] of [['ahs','#444'],['ahd','#777']]) {
    const m=document.createElementNS('http://www.w3.org/2000/svg','marker');
    m.setAttribute('id',id); m.setAttribute('markerWidth','7'); m.setAttribute('markerHeight','7');
    m.setAttribute('refX','6'); m.setAttribute('refY','3.5'); m.setAttribute('orient','auto');
    const pt=document.createElementNS('http://www.w3.org/2000/svg','path');
    pt.setAttribute('d','M0,0 L7,3.5 L0,7 z'); pt.setAttribute('fill',c);
    m.appendChild(pt); defs.appendChild(m);
  }
  DATA.edges.forEach((e, idx) => {
    const {d,pts}=edgePath(e, idx);
    const p=document.createElementNS('http://www.w3.org/2000/svg','path');
    p.setAttribute('d',d); p.setAttribute('fill','none');
    p.setAttribute('stroke', e.dash?'#777':'#444'); p.setAttribute('stroke-width','1.2');
    if(e.dash) p.setAttribute('stroke-dasharray','5 3');
    p.setAttribute('marker-end','url(#'+(e.dash?'ahd':'ahs')+')');
    eg.appendChild(p);
    if(e.label){
      let lx,ly;
      if(e.kind && e.kind.startsWith('bypass')){
        lx=pts[1][0]+(e.kind.startsWith('bypassL')?-5:5);
        ly=(pts[0][1]+pts[pts.length-1][1])/2;
      } else {
        const mid=pts[Math.floor(pts.length/2)];
        lx=mid[0]+7; ly=mid[1]-8;
      }
      const t=document.createElementNS('http://www.w3.org/2000/svg','text');
      t.setAttribute('x',lx); t.setAttribute('y',ly);
      t.setAttribute('font-size','10'); t.setAttribute('fill','#111');
      t.setAttribute('text-anchor', e.kind&&e.kind.startsWith('bypassL')?'end':'start');
      t.textContent=e.label; eg.appendChild(t);
    }
    // 连线中点手柄（拖动 = 手动调整走线, 双击 = 恢复自动走线）
    let bi = 0, bl = -1;
    for (let i = 0; i < pts.length - 1; i++) {
      const l = Math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]);
      if (l > bl) { bl = l; bi = i; }
    }
    const hnd = HANDLES[idx];
    const hx = hnd ? hnd.x : (pts[bi][0]+pts[bi+1][0])/2, hy = hnd ? hnd.y : (pts[bi][1]+pts[bi+1][1])/2;
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', hx); c.setAttribute('cy', hy); c.setAttribute('r', 4.5);
    c.setAttribute('class','hnd'); c.dataset.edge = idx;
    c.addEventListener('pointerdown', ev => {
      HANDLES[idx] = {x: hx, y: hy};  // 先在当前中点建立手柄, 拖动按增量移动
      drag = {type:'handle', idx, x:ev.clientX, y:ev.clientY, hx, hy};
      ev.target.setPointerCapture(ev.pointerId); ev.preventDefault(); ev.stopPropagation();
    });
    c.addEventListener('dblclick', () => { delete HANDLES[idx]; drawEdges(); });
    eg.appendChild(c);
  });
}
let drag=null;
function startDrag(ev){ const id=ev.currentTarget.dataset.id;
  drag={type:'node', id, dx:ev.clientX-nodes[id].x, dy:ev.clientY-nodes[id].y};
  ev.currentTarget.setPointerCapture(ev.pointerId); ev.preventDefault(); }
window.addEventListener('pointermove', ev=>{ if(!drag) return;
  const cvo = cv.getBoundingClientRect();
  if (drag.type === 'handle') {
    const h = HANDLES[drag.idx] || (HANDLES[drag.idx] = {x:drag.hx, y:drag.hy});
    h.x = drag.hx + (ev.clientX - drag.x); h.y = drag.hy + (ev.clientY - drag.y);
  } else {
    nodes[drag.id].x=ev.clientX-drag.dx; nodes[drag.id].y=ev.clientY-drag.dy;
    place(drag.id);
  }
  drawEdges(); });
window.addEventListener('pointerup', ()=> drag=null);
buildDOM();
function renderMathAndLayout(){
  if (window.renderMathInElement) {
    renderMathInElement(document.body,
      {delimiters:[{left:'$$',right:'$$',display:false},{left:'$',right:'$',display:false}]});
  }
  resetLayout();  // KaTeX 渲染改变盒子尺寸后必须重排, 否则行间通道会落进组件内
}
window.addEventListener('load', renderMathAndLayout);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(()=>resetLayout());
setTimeout(resetLayout, 500);
</script>
</body>
</html>
"""


def write_html(path):
    gqa_text = ("para", [
        "多头拆分与 GQA 分组（H=8, K=4）",
        r"查询头 $h_1 \\dots h_8$（×H=8, 每头 d 维）",
        r"KV 头 $kv_1 \\dots kv_4$（×K=4）",
        r"$N_{rep} = H \\div K = 2$: 每 2 个查询头共享 1 组 KV",
        "q_norm/k_norm/RoPE 逐头独立;",
        "scores 对 H 个头并行独立,",
        "输出 concat",
    ])
    nodes_data = {**{nid: {"shape": sh, "lines": lines}
                     for nid, (sh, lines) in NODES.items() if sh != "gqa"},
                  "GQAP": {"shape": gqa_text[0], "lines": gqa_text[1]}}
    data = {
        "nodes": nodes_data,
        "colmap": COLMAP,
        "sections": [{"title": s["title"], "rows": s["rows"], "sides": s["sides"]}
                     for s in SECTIONS],
        "edges": [{"s": s, "t": t, "dash": bool(d), "sa": list(sa) if sa else None,
                   "ta": list(ta) if ta else None, "label": label, "kind": kind}
                  for (s, t, d, sa, ta, label, kind) in EDGES
                  if s in nodes_data and t in nodes_data],
    }
    html = HTML_TMPL.replace("%%DATA%%", json.dumps(data, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="生成 MiniMind 完整网络结构图")
    ap.add_argument("--svg", default=os.path.join(repo, "docs", "network_architecture.svg"))
    ap.add_argument("--html", default=os.path.join(repo, "docs", "network_architecture.html"))
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    mfig = plt.figure(figsize=(20, 20), dpi=72)
    maxs = mfig.add_axes([0, 0, 1, 1])
    maxs.axis("off")
    boxes = measure(mfig, maxs)
    plt.close(mfig)

    W, H, titles = layout(boxes)
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(fig, ax, boxes, W, H, titles)

    for p in [args.svg] + ([args.png] if args.png else []):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(args.svg, dpi=72, facecolor="white")
    if args.png:
        fig.savefig(args.png, dpi=144, facecolor="white")
    plt.close(fig)
    write_html(args.html)
    print(f"font={FAMILY}  svg={W:.0f}x{H:.0f}pt  ->  {args.svg}")
    print(f"html -> {args.html}")


if __name__ == "__main__":
    main()
