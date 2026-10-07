# 文件导读：训练入口共用的日志、调度、种子、分布式、初始化与断点工具。
# 建议先读 init_model/get_lr，再读 lm_checkpoint 与 SkipBatchSampler，最后读奖励接口。
# token 是分词器编号，tokenizer 把文本与编号互转；模型配置决定词表及隐藏层的张量尺寸。
# epoch 表示一次数据遍历，step 在调用方多指数据批次；它不必等于 optimizer 的更新次数。
# 权重文件只描述参数；完整训练续跑还要优化器、步数、缩放器等状态。
# DDP（Distributed Data Parallel，分布式数据并行）每进程持一份模型，反向时同步梯度。
# 部分路径以 trainer 工作目录为基准；奖励模型需已下载，trust_remote_code 会执行模型仓库代码。
"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import operator
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

# 打印总参数量与估计活跃参数量，单位百万；不返回计数对象。
# 根据名字识别第 0 专家并乘专家数，假设各专家等宽；不是逐 token 实测 FLOPs。
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 仅名字包含 mlp.experts.0. 的参数计入单专家模板容量。
    # 权重共享按 model.parameters() 去重，避免把同一 Parameter 数两遍。
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    # 仅名字包含 mlp.shared_experts.0. 的参数计入单共享专家容量。
    # MiniMind 当前没有共享专家，这一项为 0；兼容其他模型时假设共享专家等宽。
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


# 未初始化分布式或 rank=0 时返回 True；rank 是全局进程编号。
# 用于避免多进程重复日志/保存，不表示其他进程停止训练。
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 接收可打印 content，仅主进程 print；不写日志文件，也无返回值。
def Logger(content):
    if is_main_process():
        print(content)


# 返回余弦衰减 lr*(0.1+0.45*(1+cos(pi*step/total)))。
# step=0 得 lr，step=total 得 0.1*lr；无 warmup，total_steps 须非零。
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


# 以环境 RANK 判断 torchrun 模式，初始化 NCCL 并选 LOCAL_RANK 对应 GPU。
# 返回本机卡号；单进程返回 0，不创建进程组；NCCL 路径要求 CUDA。
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置 Python、NumPy、PyTorch 与 CUDA 随机种子，并切换 cuDNN 确定性选项。
# 影响初始化、打乱及采样；不同设备/算子/并行次序仍可能有数值差异。
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 双模式：model 非空时保存权重及 _resume.pth，空时尝试加载后返回字典/None。
# 保存优化器、epoch/step、world_size、记录平台 id 及 kwargs 中额外状态。
# 未保存全部 RNG（随机数生成器）状态、采样引擎状态或未更新的累积梯度，
# 所以“可续训”不等于跨任意保存点逐位复现下一步。
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 先写同目录临时文件再 os.replace，避免读取半写文件；两份文件不是整体事务。
        # 权重转 CPU float16 会舍入，resume 的 model 也使用该半精度字典。
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            # AdamW（解耦权重衰减的自适应优化器）除权重还维护一/二阶矩与更新计数。
            # 只恢复权重会重置这些历史，不能等同断点续训。
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # 按旧/新 world_size 折算批次步号，使用整数除法截断。
            # 这只近似对齐已消费数据量，不保证不同并行度下样本顺序完全一致。
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


# 按配置创建原生模型并加载 from_weight 前缀权重，返回 (设备上的模型,分词器)。
# none 跳过加载；strict=False 允许缺/多键，但同名张量尺寸不符仍可报错。
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


# 将索引流组为批次，跳过已完成批次数，使续训不重新消费前部数据。
# 只跳索引，不恢复 Python/worker 中随机增强的内部状态。
class SkipBatchSampler(Sampler):
    # sampler 可是分布式采样器或索引列表；记录批大小与 skip_batches。
    # 调用方须保证 batch_size>0；对象不读取样本内容。
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    # 顺序收集索引，先丢弃指定批次，后 yield 索引列表。
    # 尾批不足 batch_size 也产出，没有 drop_last；迭代改变局部计数不改原数据。
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    # 向上取整得到总批数，再减跳过数并截至非负。
    # 例如 10 条、每批 4、跳 1，剩 2 批（4 条及 2 条）。
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# 奖励模型适配器，将对话与回答映射为标量质量分。
# 它不是策略网络；只提供评分，无反向更新接口。
class LMForRewardModel:
    # 加载专用 tokenizer/AutoModel，迁移设备并 eval。
    # 要求该远程模型实现 get_score；不是任意 AutoModel 都符合此协议。
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    # 无梯度评分：合并旧对话为文本，再构造 user/assistant 两条消息。
    # 返回裁剪到 [-3,3] 的标量；这是防极端奖励的规则，不是概率或准确率。
    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ===== 数学表达式安全求值：替代 eval，只放行算术运算与 math 白名单（长度上限 512） =====
# 把表达式解析为 AST（抽象语法树），递归解释白名单运算而非 Python eval。
# 返回数值或抛异常；长度上限 512，拒绝字符串、任意属性及非白名单调用。
# 限制幂规模降低资源耗尽风险，但不是完整操作系统级沙箱。
def safe_math_eval(expression):
    """对模型生成的数学表达式求值：支持 + - * / // % **、math 白名单函数与 pi/e/tau 常量。"""
    # 对 base/exp 限制指数绝对值及粗估结果十进制规模，再执行 **。
    # 过大幂抛 ValueError；负底数/分数幂等仍服从 Python 数值规则。
    def pow_guard(base, exp):  # 幂运算统一走这里，拦截 9**9**9 / 10**99999 这类把进程算死的输入
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp
    # 返回裸名字或 math.name 的 name；其他属性访问返回 None。
    # 只做名称提取，后续须再命中 funcs/consts 才可求值。
    def resolve(node):  # 同时兼容 sqrt(4) 与 math.sqrt(4) 两种写法
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr
    # 递归处理数字、常量、单目/双目运算及无关键字函数调用。
    # 运算符以类型查表；未知名称/运算可抛 KeyError，语法未支持则 ValueError。
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    # 函数表限制可调用名称；pow 特别走规模检查，其他函数仍可能因输入域错误抛异常。
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        raise ValueError('不支持的表达式语法') from None
