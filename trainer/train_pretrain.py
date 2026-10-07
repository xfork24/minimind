# 文件导读：预训练入口，从普通 text JSONL 学习“前文→下一个 token”。
# 默认从随机参数开始（from_weight=none），输出 pretrain_<宽度>.pth。
# 运行前提：从 trainer 目录执行，准备匹配的 JSONL、分词器与起始权重。
# 阅读顺序：先主入口初始化，再 train_epoch 前向损失、反向更新、保存与恢复。
# token 是分词器编号；输入 [B,L]，logits 是 [B,L,V] 的未归一化词表分数。
# 交叉熵（Cross Entropy，CE）使目标 token 更可能；-100 标签忽略直接监督，仍保留输入上下文。
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
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# 执行一个 epoch 的微批循环，输入 loader、总批数 iters 与已完成 start_step。
# 从模块全局读取模型、优化器、args 与精度上下文；返回 None，副作用是更新与保存。
# 仅累积组末尾/epoch 尾部更新，日志与保存另按步数触发。
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 每个微批按全局批次进度更新学习率；与真正 optimizer 更新频率不完全相同。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 模型内将 logits 去尾、labels 去首；预训练 labels 覆盖非 pad 的文本与边界。
            # 交叉熵为 -log p(目标 token)，概率越低损失越大。
            res = model(input_ids, labels=labels)
            # 总目标=下一 token 交叉熵+MoE 路由辅助项；稠密模型辅助项为零。
            loss = res.loss + res.aux_loss
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
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

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

        del input_ids, labels, res, loss


if __name__ == "__main__":
    # 命令行参数分四组：数据/权重路径，模型形状，优化超参数，设备/日志/续训。
    # 默认值只描述本脚本选项，不推导显存需求，也不校验数据可用性。
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # 解析选项，不加载模型；真正副作用从后面的目录、设备与权重初始化开始。
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # torchrun 环境用 LOCAL_RANK 选本机 GPU；全局 rank 决定主日志进程。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
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
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers 是加载数据的子进程数；pin_memory 便于 CPU→CUDA 传输，不把数据直接留在 GPU。
        # batch_sampler 已决定顺序、分批和跳批，不另设置 batch_size 或 shuffle。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 所有进程同步后销毁通信组；不能只让主进程进入 barrier。
        dist.barrier()
        dist.destroy_process_group()