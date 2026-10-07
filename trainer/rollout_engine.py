# 文件导读：统一 PyTorch 本地生成与 SGLang HTTP 推理的 rollout 接口。
# 输入提示 ID/遮罩，输出完整轨迹、回答、旧对数概率、各行提示长度及有效回答遮罩。
# 无策略优化器；训练脚本负责奖励/损失，update_policy 负责同步采样权重。
# rollout（策略采样轨迹）记录模型生成的 token（分词器编号）；logp 是对应动作的自然对数概率。
# logits 是词表分数，log_softmax 转为 logp；损失遮罩与 attention_mask（注意力可见性）用途不同。
# EOS（End of Sequence）表示生成结束，PAD（Padding）补齐各行；二者编号由 tokenizer 定义。
# B=提示数，G=采样数，P=提示宽度，R=回答宽度；展平后 N=B*G。
# 优先读 RolloutResult、TorchRolloutEngine，再读服务协议与权重同步。
"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
# 取最后 n_keep 个目标 token 的 log p，输入 [N,L]，返回 [N,n_keep]。
# 尾部 n_keep+1 个 logits 去最后一步，才能用前一位置预测对应 ID；n_keep≤0 返回空张量。
# 函数本身不关闭梯度；调用方的 no_grad 决定是否建图，DDP 在这里解包绕过同步包装。
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            # log_softmax 后按目标 ID gather：[R,V] 配索引 [R,1] 得 [R,1]，挤掉末维得 [R]。
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
# 数据类只组织结果，不校验长度。output_ids=[N,P+R]，completion_ids/logps/mask=[N,R]。
# prompt_lens=[N] 定位各行回答起点，completions 是 N 个解码字符串。
# 远程后端可有可变提示长度；不能总把最大 P 当每行真实长度。
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
# 抽象接口：rollout 生成样本，update_policy 同步参数；不可直接实例化。
# 统一结构不代表各后端温度、停止或 logprob 定义完全一致。
class RolloutEngine(ABC):
    tokenizer = None
    
    # 约定输入 [B,P] 与对应遮罩，每个提示生成 G 条回答，返回 RolloutResult。
    # 抽象方法不实现采样；子类负责保持“同提示连续 G 条”的顺序。
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass
    
    # 约定以当前训练模型更新采样策略；本方法只有接口，无参数复制行为。
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
# 本地后端复用模型对象；采样与重算 logp 在 no_grad 下进行。
# 不启动网络服务，也不持有独立优化器。
class TorchRolloutEngine(RolloutEngine):
    # 保存策略、分词器、设备及可选自动混合精度上下文的引用，不复制参数。
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    # 重复提示、调用 generate，再以完整序列重新计算原始模型 logp。
    # 输出不带梯度；返回的 completion_mask 全为 1，EOS 后屏蔽由下游训练代码完成。
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        # no_grad 不改变 train/eval 状态；此处不自动 eval，非零 dropout 时采样与重算可能有随机差异。
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                # 沿 batch 连续重复：两条提示、G=2 时为 a,a,b,b，奖励分组依赖这个次序。
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            # 重算的是未经温度缩放/采样截断的模型条件概率，不自动等于采样器的实际概率。
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    # 把 self.policy_model 指向传入模型；不是 deepcopy，也不生成新权重快照。
    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
# 远程生成后端，通过 requests 模块调用服务；训练与采样可以在不同进程。
# 权重重载使用磁盘路径，训练进程与服务须能访问同一目录。
class SGLangRolloutEngine(RolloutEngine):
    # 保存 URL、共享 checkpoint 路径与超时，并从模型目录加载 tokenizer。
    # 分词器必须与服务端一致，否则 token 编号及解码语义不匹配。
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    # 去掉提示 padding，按提示连续重复 G 次，请求 /generate 与 output_token_logprobs。
    # 解析回答后右补齐张量，返回各行原提示长度；不对服务内部采样建立梯度图。
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            # 服务少返回 logp 时前补 0，多返回时保留尾部；补值只是形状兼容，不是正确概率。
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        # 非空结果列表至少补到 1 个回答位置；空列表仍会在内层 max 处报错。
        # mask 为零表示补齐位置非真实动作，不把补齐的 logp=0 当作已采样动作。
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        # 闭包读取 device，把不等长 Python 列表右补至 max_len，再构造张量。
        # ID 通常推断为整数，含 0.0 的 logp 推断为浮点；max_len 须覆盖最长行。
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    # 主 rank 解包并导出 CPU 半精度 Transformers 权重，再请求服务磁盘重载。
    # 广播成功标志并 barrier，失败在各 rank 抛异常；返回布尔成功状态。
    # 不是通过 HTTP 上传权重文件，共享文件系统是前提。
    def update_policy(self, model: torch.nn.Module):
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    # 请求清除服务推理缓存，返回 HTTP 是否为 200；网络异常在此不捕获。
    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    # 短超时 GET 健康接口，异常或非 200 返回 False；不改变策略。
    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
# 根据 torch/sglang 分派构造后端，返回接口对象；未知类型抛 ValueError。
# 仅为 torch 传策略引用，为 sglang 传服务 URL/模型/共享路径。
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
