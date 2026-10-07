# MiniMind 全源码中文教学注释计划

## 目标与范围

为仓库全部 **22 个 Python 源文件**添加详细的简体中文教学注释，覆盖根目录及所有源码子目录，包括两个空的 `__init__.py`。

读者定位为有工程经验、缺乏大模型背景的工程师。每个文件独立可读：术语在该文件首次出现时解释中文名称、必要的英文名称、背景及用途，允许适度重复。

保留所有原有代码和注释，只插入独立的 `#` 注释行。公共接口、函数签名、配置值和运行逻辑保持原样；保留 JSON、文档、依赖文件及内嵌聊天模板原样，在对应源码旁解释其作用。

## 注释组织与深度

- **文件导读**：介绍模块职责、上下游关系、输入输出、运行前提及建议阅读顺序；空的包初始化文件解释 Python 包的作用。
- **类与函数说明**：覆盖全部类、函数、方法及嵌套函数，解释职责、参数、返回值、状态变化、梯度关系和关键边界条件。
- **关键代码块说明**：解释操作目的、执行过程和设计理由。复杂表达式、索引、广播、张量变形及原地操作逐项讲解；普通流程采用连贯的分块注释。
- **背景与实例**：首次遇到概念时补充必要基础；在易混淆处加入小型维度示例、概率示例或公式，并把公式变量对应到代码。
- **准确性要求**：以实际实现为准；区分通用原理、本仓库实现及适用条件。疑难概念核对官方文档或原始论文，发现现有问题也不修改代码、不把理想行为写成实际行为。

## 各模块教学重点

| 模块 | 重点内容 |
|---|---|
| 模型与 LoRA | 自回归预测、词嵌入、注意力中的 Q/K/V、分组查询注意力、因果遮罩、RMSNorm、RoPE、门控前馈网络、残差连接、混合专家及辅助损失、权重共享、KV 缓存、低秩适配与合并 |
| 数据与分词器 | token 与字符的区别、字节级 BPE、特殊 token、聊天模板、各阶段 JSONL 格式、截断与填充、标签移位、`-100` 忽略标签、回答区域监督及不同遮罩的用途 |
| 基础训练与公共工具 | 前向传播与反向传播、交叉熵、AdamW、学习率调度、梯度累积与裁剪、混合精度、随机种子、分布式训练、采样器、权重保存与完整断点恢复的区别 |
| 蒸馏与强化学习 | 教师与学生、软标签及温度、KL 散度方向、DPO 偏好对、策略与参考模型、奖励与优势、PPO 价值网络与 GAE、GRPO 组内归一化、CISPO、采样与训练策略同步、多轮工具调用及参与损失的 token |
| 推理与应用脚本 | 训练与推理模式、概率采样、temperature/top-k/top-p、结束条件、模型格式转换、工具协议、流式响应、线程与队列、API 请求响应、Web 会话状态及模型加载缓存 |

统一在相关文件中定义张量维度符号，如批大小、序列长度、隐藏维度、词表大小和注意力头数；明确每次重要形状变换及其用途。

## 实施顺序

1. 保存修改前的源码字节基线，确认完整文件清单及现有工作区改动。
2. 按模型与数据、公共工具与基础训练、蒸馏与强化学习、推理与应用的顺序逐文件添加注释。
3. 每批完成后检查内容保留和语义一致性，再审阅跨模块概念、维度和数据流是否一致。
4. 最终逐文件检查覆盖情况，交付修改文件清单及验证结果。

不新增独立教程或重构代码；教学内容直接放在对应源文件中。现有 `AGENTS.md` 和其他用户改动保持原样。

## 验证与验收

- **完整保留**：对照字节基线，确认所有原始行及其顺序保持不变，差异只有新增的注释行。
- **词法一致**：排除注释和非语义换行后，比较 token 类型与文本，确保字符串、文档字符串、缩进和有效代码完全一致。
- **结构一致**：比较忽略位置属性的 AST，并对所有文件执行编译检查，不执行训练、下载或服务启动。
- **覆盖完整**：22 个文件都有导读；全部类和函数都有说明；核心计算、配置、主流程及关键边界均得到解释。
- **教学审阅**：确认新增说明为简体中文、首次术语解释充分、维度与公式符合实现，避免仅翻译代码或作无依据的推断。

注释会改变源码行号；上述检查保证有效代码与结构不变，不要求报错行号保持原值。


## 执行完成记录（2026-10-07）

状态：全部完成。接续保留已有工作，并审阅修正新增注释中的不准确说明；最终相对执行前基线新增 **1,133 行独立中文注释**，覆盖 **22 个文件、169 个类/函数/方法定义（含嵌套定义）**。

- [x] 找回原始字节基线 `/private/tmp/minimind-zh-notes/baseline.json`，确认与下述固定提交一致。
- [x] 全部源码完成文件导读、类/函数说明、关键计算、配置及主流程解释。
- [x] 完成跨模块审阅：维度、标签移位、监督/注意力遮罩、工具动作、概率/梯度、采样同步。
- [x] 原始行字节及顺序完整保留；仅插入独立 `#` 注释行，原末尾换行状态也保留。
- [x] 有效 token（含字符串、缩进）及忽略位置属性的 AST 完全一致；未向多行字符串内部插入注释。
- [x] 全部文件有导读，全部定义在声明或装饰器之前有新增说明。
- [x] Python 3.10.20 下逐文件编译、`compileall` 和 `git diff --check` 全部通过。

既有 `.python-version`、`AGENTS.md` 与暂存状态保留；未提交、未回滚。按计划只验证语义保留，不执行训练、下载、推理或服务启动。

审阅修正了新增注释对工具声明缺失、评测脚本 LoRA 分支、共享专家统计和版本分支的误述；补充 SFT/DPO 扫描未单独排除 PAD、原生/远程采样 logp 差异、工具观察不参与策略损失、Web `max_length` 对自定义生成接口的边界。所有说明以现有实现为准，未修改这些行为。

疑难概念核对：[PyTorch KL 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html)的对数输入、概率目标与 `batchmean` 约定；[PyTorch SDPA 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)的显式 dropout 控制；[DPO 原论文](https://arxiv.org/abs/2305.18290)的偏好目标；[MiniMax-M1 原文](https://arxiv.org/html/2506.13585v1)的 CISPO 重要性权重裁剪。源码注明了本仓库 CISPO 的 KL 项、上界参数和归约方式与论文的差异。

### 完成文件清单

| 文件 | 新增注释行 | 类/函数/方法数 |
|---|---:|---:|
| `dataset/__init__.py` | 3 | 0 |
| `dataset/lm_dataset.py` | 77 | 27 |
| `eval_llm.py` | 30 | 2 |
| `model/__init__.py` | 3 | 0 |
| `model/model_lora.py` | 28 | 8 |
| `model/model_minimind.py` | 147 | 29 |
| `scripts/chat_api.py` | 14 | 0 |
| `scripts/convert_model.py` | 33 | 6 |
| `scripts/eval_toolcall.py` | 42 | 9 |
| `scripts/serve_openai_api.py` | 45 | 10 |
| `scripts/web_demo.py` | 52 | 13 |
| `trainer/rollout_engine.py` | 47 | 17 |
| `trainer/train_agent.py` | 90 | 9 |
| `trainer/train_distillation.py` | 70 | 2 |
| `trainer/train_dpo.py` | 66 | 3 |
| `trainer/train_full_sft.py` | 54 | 1 |
| `trainer/train_grpo.py` | 67 | 3 |
| `trainer/train_lora.py` | 55 | 1 |
| `trainer/train_ppo.py` | 70 | 6 |
| `trainer/train_pretrain.py` | 53 | 1 |
| `trainer/train_tokenizer.py` | 29 | 3 |
| `trainer/trainer_utils.py` | 58 | 19 |

### 复验命令

从仓库根目录执行。原校验脚本和基线在 `/private/tmp/minimind-zh-notes/`；以下独立命令读取固定基线提交，临时目录清理后仍可复验。它不导入或执行业务模块。

```sh
PYTHONPYCACHEPREFIX=/tmp/minimind-notes-compile .venv/bin/python -m compileall -q model trainer dataset scripts eval_llm.py
git diff --check
.venv/bin/python - <<'PY'
import ast,io,json,pathlib,subprocess,tokenize
root=pathlib.Path.cwd()
baseline_commit='f659b55761b754d306bd140573493a6543cafd7f'
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline_commit],text=True).splitlines()
base={p:subprocess.check_output(['git','show',f'{baseline_commit}:{p}']).decode('utf8') for p in paths if p.endswith('.py')}
assert len(base)==22
files=set(subprocess.check_output(['rg','--files','-g','*.py'],cwd=root,text=True).splitlines())
assert files==set(base),(files-set(base),set(base)-files)
ignored={tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENCODING,tokenize.ENDMARKER}
def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type not in ignored]
total=0
for path,old in sorted(base.items()):
    raw=(root/path).read_bytes(); new=raw.decode('utf8')
    assert raw==new.encode('utf8')
    orig=old.splitlines(keepends=True); current=new.splitlines(keepends=True)
    i=0; added=[]
    for n,line in enumerate(current,1):
        if i<len(orig) and line==orig[i]: i+=1
        else:
            assert line.lstrip().startswith('#'),(path,n,'not comment-only insertion')
            added.append(n)
    assert i==len(orig),(path,'original lines missing')
    assert tokens(old)==tokens(new),(path,'token mismatch')
    assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False),(path,'AST mismatch')
    commentlines={t.start[0] for t in tokenize.generate_tokens(io.StringIO(new).readline) if t.type==tokenize.COMMENT}
    assert set(added)<=commentlines,(path,'inserted inside multiline string')
    assert '文件导读' in current[0],(path,'missing header')
    count=0
    for node in ast.walk(ast.parse(new)):
        if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            line=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
            assert line-1 in added,(path,node.name,'missing adjacent teaching note')
            count+=1
    compile(new,path,'exec')
    total+=len(added)
    print(f'{path}: +{len(added)} comments, {count} definitions, lines/tokens/AST/compile OK')
print(f'PASS: {len(base)} files, {total} inserted comment lines')
PY
```
