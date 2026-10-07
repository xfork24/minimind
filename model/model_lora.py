# 文件导读：低秩适配（Low-Rank Adaptation，LoRA）在已有线性层旁增加可训练支路。
# 基矩阵 W 不必整体微调，改变量 ΔW=B*A，秩 r 远小于输入/输出宽度时节省训练参数。
# 本实现无额外 alpha/r 缩放，仅自动处理输入、输出维度相等的 Linear。
# 上游加载基模，apply_lora 注入；trainer/train_lora.py 冻结基模并训练，推理可加载或合并。
# 阅读顺序：LoRA → apply_lora → 保存/加载 → merge_lora。
# 磁盘仅适配器权重不足以独立推理，必须配原基模、注入位置与秩。
# 例如 in=out=768、r=16，基层有 768² 个权重，支路仅有 2*768*16 个。
# “低秩”限制的是增量 BA 的秩，不表示原矩阵 W 本身必须低秩。
import torch
from torch import optim, nn


# 定义Lora网络结构
# 表示 Δy=B(A(x)) 的小型模块；参数矩阵 A=[r,in]，B=[out,r]。
# 输入 [...,in]，输出 [...,out]，与基层输出相加；不是把输入转成离散低秩编号。
class LoRA(nn.Module):
    # 保存秩并创建两个无偏置投影；A 随机、B 全零，让初始增量输出为零。
    # 首次反向 B 可获得梯度，A 的梯度受零 B 影响；后续 B 更新后两者可共同学习。
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    # 先压到 rank 再投回 out_features，返回增量而非基层完整结果。
    # 本方法不自动冻结任何参数，是否训练基模由调用脚本控制。
    def forward(self, x):
        return self.B(self.A(x))


# 遍历 model，给方形线性层挂 lora 属性并替换 forward；原地修改模型，无返回值。
# 默认 rank=16，适配器放到 model.device；这里只指定设备，未显式对齐基层 dtype。
# 非方形 Q/K/V 或前馈投影不会仅因名字匹配而被注入，选择条件完全是尺寸。
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            # 返回原层结果+低秩增量；默认参数绑定本轮 layer1/layer2。
            # 这样避免闭包晚绑定让所有层都引用循环最后一个适配器；输出仍同形状。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


# 从 path 读取适配器字典，去掉 DDP 的 module. 前缀，分层加载 A/B。
# 要求模型已 apply_lora，且秩、层名一致；不创建缺失支路，也不加载基模权重。
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


# 只保存带 lora 属性模块的参数，转 CPU float16 以减少文件体积。
# 剥离编译包装及层名 module. 前缀；不保存优化器、随机状态或训练进度。
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


# 先加载适配器，再导出 W+BA 的基模格式权重到 save_path。
# 生成文件排除 .lora. 键，不在运行中的模型上反复叠加增量。
# 合并用 CPU 半精度，存在舍入误差；要求调用前已建立相同适配器结构。
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # PyTorch Linear 存储 [out,in]，故 [out,r]@[r,in] 与 W 对齐。
                # 此处 += 修改导出字典的副本；不用转置，且没有 alpha/r 系数。
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
