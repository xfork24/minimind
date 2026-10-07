# 文件导读：知识蒸馏（Knowledge Distillation）用冻结教师的分布指导学生。
# 硬标签为样本目标 token，软标签为教师对全部词表类别的概率。
# logits 是词表上的未归一化分数；softmax 转为概率，log_softmax 给自然对数概率。
# KL（Kullback–Leibler divergence，库尔贝克–莱布勒散度）衡量两个分布的差异，通常不对称。
# 此处 KL(P_teacher||P_student)=sum_v P_teacher(v)*log(P_teacher(v)/P_student(v))。
# 仅学生更新；此实现假设教师、学生 token 编号语义一致，不能只靠截词表完成对齐。
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
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# 输入 [N,V] 学生/教师 logits；温度 T 先除 logits，再 softmax/log_softmax。
# 返回 T²*KL(教师||学生)，不是 KL(学生||教师)；教师 no_grad/detach，学生保留梯度。
# 默认 batchmean 把所有类别 KL 求和再除 N（此调用 N 是有效回答 token 数）。
# 温度越高分布越平缓，可暴露次选类别关系；T² 补偿温度对梯度尺度的影响。
# temperature 须为正且两者类别对应一致；T² 不保证所有温度下梯度完全相同。
# 接口核对：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


# 执行一个 epoch 的微批循环，输入 loader、总批数 iters 与已完成 start_step。
# 从模块全局读取模型、优化器、args 与精度上下文；返回 None，副作用是更新与保存。
# 仅累积组末尾/epoch 尾部更新，日志与保存另按步数触发。
# teacher_model 可为空；否则以冻结教师分布与学生做回答 token 蒸馏。
def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()
    
    if teacher_model is not None:
        teacher_model.eval()
        # eval 关 dropout，requires_grad_(False) 冻结参数，no_grad 进一步省去前向图。
        # 教师不被学生 optimizer 持有。
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 目标已向后移一位，mask=[B,L-1]，仅挑有效回答位置做两种监督。
        loss_mask = (labels[..., 1:] != -100).float()
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 每个微批按全局批次进度更新学习率；与真正 optimizer 更新频率不完全相同。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                # 仅截取教师词表前 V_student 维；前提是两者共享该范围内的 token 映射。
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        # 逐位置交叉熵乘 mask，再除有效位置数；+1e-8 防零分母。
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss
        if teacher_model is not None:
            distill_loss = distillation_loss(
                # 布尔索引把 [B*(L-1),V] 变为 [N有效,V]，只蒸馏回答区域。
                # 若 N有效=0，KL 的 batchmean 仍可能产生非有限值，未特别兜底。
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                # 教师采用同一有效位置索引，确保每行软目标对应学生正在预测的同一个位置。
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        # alpha 控硬标签比例，1-alpha 控软分布；MoE 辅助项在 ce_loss 内也乘 alpha。
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

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
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 解开 DDP/compile 包装，导出无前缀权重；eval/train 控制 dropout，不负责关闭梯度。
            state_dict = raw_model.state_dict()
            # out 保存推理用半精度权重；随后 checkpoints 另存优化器/缩放器/进度。
            # 两者目的不同，权重文件不足以恢复优化器历史。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    # 命令行参数分四组：数据/权重路径，模型形状，优化超参数，设备/日志/续训。
    # 默认值只描述本脚本选项，不推导显存需求，也不校验数据可用性。
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
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
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    # 仅 from_resume=1 时尝试读完整断点；from_weight 只控制起始参数来源。
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
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
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    # eval 关 dropout，requires_grad_(False) 冻结参数，no_grad 进一步省去前向图。
    # 教师不被学生 optimizer 持有。
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        # 所有进程同步后销毁通信组；不能只让主进程进入 barrier。
        dist.barrier()
        dist.destroy_process_group()