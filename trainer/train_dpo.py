# 文件导读：直接偏好优化（Direct Preference Optimization，DPO）从好/坏回答对学习。
# 策略模型可训练，参考模型冻结；比较二者对 chosen/rejected 的相对对数概率。
# 不现场采样，不需要独立奖励网络；输入是 DPODataset 已移位的 x/y/mask。
# logits 是未归一化词表分数，log_softmax 将其转为自然对数概率；log p 越大越偏好该序列。
# chosen/rejected 的条件提示应一致；参考模型记录起始行为，帮助控制偏离原模型的程度。
# 运行前提：从 trainer 目录执行，准备匹配的 JSONL、分词器与起始权重。
# 阅读顺序：先主入口初始化，再 train_epoch 前向损失、反向更新、保存与恢复。
# token 是分词器编号；每对有两条序列，拼批后输入 [2B,L]，logits 为 [2B,L,V]。
# 本路径用 0/1 回答 mask 聚合 logp；y 中始终保留合法编号，不使用 SFT 的 -100 标签。
# MoE（Mixture of Experts，混合专家）每 token 选少量前馈网络，辅助项帮助分散路由负载。
# B=微批大小，L=序列长度，V=词表大小；一次更新可累积多个微批的梯度。
# 梯度是损失对参数的导数，backward 只累加导数，optimizer.step 才改变参数。
# 自动混合精度（AMP）在合适算子用半精度；float16 配 GradScaler 缓解梯度下溢。
# 分布式数据并行（DDP）同步各进程梯度；有效批量约为 B*累积步数*进程数。
import os
import sys

# 显式补包上下文及仓库根搜索路径，保持直接运行脚本的相对导入约定。
# 这不改变数据路径基准，../dataset 仍相对于当前工作目录。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# logits=[B,L,V]，labels=[B,L]；log_softmax 后沿词表轴 gather 目标 token。
# 返回 [B,L] 对数概率，非整词表分布；labels 必须是有效编号而非 -100。
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


# 输入前半 chosen、后半 rejected 的 [2B,L] 对数概率与 mask。
# 本实现按回答 token 求和（不是长度平均），得各序列 log p；长回答会影响总和尺度。
# 返回 mean(-log sigmoid(beta*((logπ好-logπ坏)-(logref好-logref坏))))。
# 参考分布无梯度，策略提升相对偏好；beta 调对参考偏移的尺度，非学习率。
# sigmoid(u)=1/(1+exp(-u))，把相对偏好差转为“好回答胜出”的概率；beta 通常取正值。
# 原论文：https://arxiv.org/abs/2305.18290
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # 屏蔽用户与填充位置，沿 L 求和；对数概率相加相当于条件概率连乘。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 例如策略的好/坏概率比为 4、参考为 2，则 logits=log(4/2)>0，正 beta 下惩罚减小。
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


# 执行一个 epoch 的微批循环，输入 loader、总批数 iters 与已完成 start_step。
# 从模块全局读取模型、优化器、args 与精度上下文；返回 None，副作用是更新与保存。
# 仅累积组末尾/epoch 尾部更新，日志与保存另按步数触发。
# ref_model 提供无梯度基准；同一偏好对的两条序列一起前向。
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 顺序先 B 条 chosen 后 B 条 rejected，y/mask 同序；损失分半依赖此布局。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 每个微批按全局批次进度更新学习率；与真正 optimizer 更新频率不完全相同。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            # 除累积步数，使多个微批梯度相加近似大批均值。
            # 尾部不足一组仍按固定除数缩放，并非按尾组实际大小重新归一化。
            loss = loss / args.accumulation_steps

        # 放大损失后反向计算梯度；不会在这里更新参数。
        # 累积期间未 zero_grad，前面微批梯度继续留在 .grad 中。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # 先撤销缩放再裁剪，否则阈值作用于放大梯度，实际意义错误。
            # 梯度裁剪按所有待更新参数梯度的总范数统一缩放，防止单次更新过大；不裁剪参数值。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 缩放器检查非有限梯度并决定是否调用优化器；update 调整后续缩放倍率。
            scaler.step(optimizer)
            scaler.update()
            # 更新后把梯度置 None，开启下一累积组；不是把参数重置为零。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            # 日志多为当前微批损失，不是整个 epoch 的加权均值；ETA 按平均批耗时估算。
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 解开 DDP/compile 包装，导出无前缀权重；eval/train 控制 dropout，不负责关闭梯度。
            state_dict = raw_model.state_dict()
            # out 保存推理用半精度权重；随后 checkpoints 另存优化器/缩放器/进度。
            # 两者目的不同，权重文件不足以恢复优化器历史。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # 命令行参数分四组：数据/权重路径，模型形状，优化超参数，设备/日志/续训。
    # 默认值只描述本脚本选项，不推导显存需求，也不校验数据可用性。
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # 解析选项，不加载模型；真正副作用从后面的目录、设备与权重初始化开始。
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # torchrun 环境用 LOCAL_RANK 选本机 GPU；全局 rank 决定主日志进程。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 仅 from_resume=1 时尝试读完整断点；from_weight 只控制起始参数来源。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 路径为空上下文；GPU 按 dtype 自动选择算子精度。
    # bfloat16 指数范围较大，此处只对 float16 启用梯度缩放。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 变量名沿用 wandb，但实际日志后端为 SwanLab；只有主进程建立 run。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    # 参考模型始终取起始权重，不随策略更新；提供固定相对概率基准。
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡切分样本索引；单卡稍后用 randperm 洗牌。
    # DistributedSampler 可补齐索引使各卡等长，可能重复少量尾样本。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 是带解耦权重衰减的 Adam 优化器：历史梯度均值/平方均值调节各参数的更新尺度。
    # AdamW 维护梯度一阶/二阶矩，并把权重衰减与梯度更新分离。
    # parameters 中仅 requires_grad=True 且得到梯度的参数实际更新。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    # 恢复参数与调用方保存的训练状态，继续 epoch/step。
    # 未保存全部 RNG 状态及未提交累积梯度，不能承诺逐位复现中断前轨迹。
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # 编译图以优化执行，不改损失定义；保存前用 _orig_mod 取原模块。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 按 epoch 重设采样种子并跳过已完成批次，len(loader)+skip 恢复原进度标尺。
    # 这是索引层跳过，不是重新计算跳过批的梯度。
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers 是加载数据的子进程数；pin_memory 便于 CPU→CUDA 传输，不把数据直接留在 GPU。
        # batch_sampler 已决定顺序、分批和跳批，不另设置 batch_size 或 shuffle。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 所有进程同步后销毁通信组；不能只让主进程进入 barrier。
        dist.barrier()
        dist.destroy_process_group()