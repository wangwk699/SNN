# SNN Rotation 回归检查与 fast-hadamard-transform 可复现性修改说明

> **用途**：本文件用于指导部署在服务器上的 Codex 在**没有任何先验上下文**的情况下，对 `SNN` 项目完成本轮修改。  
> **项目根目录**：默认 `~/SNN`。  
> **目标仓库**：`wangwk699/SNN`。  
> **原则**：只完成本文明确要求的修改；不要顺带重构无关代码，不要改变现有 Prefix、Calibration、10-site、ANN training、SNN conversion、TL;DR/lm-eval 评估协议。

---

## 0. 本轮修改目标

本轮只完成以下三类工作：

1. **规范 `fast-hadamard-transform` 的使用和版本管理**
   - 服务器上 `~/SNN/fast-hadamard-transform/` 已经存在且不为空。
   - 用户已经在 `snn2` 环境执行：
     ```bash
     cd ~/SNN
     python -m pip install -e ./fast-hadamard-transform --no-build-isolation
     ```
   - 用户已验证 Python 实际导入路径为：
     ```text
     /home/wangwenkang/SNN/fast-hadamard-transform/fast_hadamard_transform/__init__.py
     ```
   - 用户已经按此前方案修改 `snn2/hadamard.py` 中 `_fast_fht()`：
     - CPU 可以使用纯 Torch fallback；
     - CUDA 上必须调用 `fast_hadamard_transform.hadamard_transform`；
     - CUDA import 或 runtime 失败时必须报错，**禁止 silent fallback**。
   - 本轮不要撤销上述修改。
   - 将 editable install 和验证方法写入 `环境配置.md`。
   - 同时把 `.gitmodules` 与 submodule pinned commit 规范化，使未来全新 clone 可以完整复现。

2. **增加 Rotation model-level regression check**
   - 在已有的 **128 个 calibration samples** 上比较：
     ```text
     Original Base model logits
     vs
     完整 Rotation 后模型 logits
     ```
   - 完整 Rotation 必须包含：
     - fused/offline rotation；
     - online R3；
     - online R4；
     - `SiteController(mode="identity")`；
     - 不做任何 activation replacement；
     - 不使用 Prefix。
   - 目的仅为验证 Rotation 是 function-preserving transformation：
     \[
     f_{\mathrm{Base}}(x) \approx f_{\mathrm{Rotated}}(x)
     \]
   - **不要增加不同 seed 的 absolute-value distribution test。**

3. **更新 `实验执行总结.md`**
   - `prepare_rotation.py` 的说明必须补充：运行期间会自动使用已有 calibration manifest 中的 128 个样本执行 Base ↔ Rotated logits regression check。
   - 用户后续会严格按照 `实验执行总结.md` 运行实验，所以命令顺序和失败条件必须写清楚。

---

# 1. 修改前必须先检查仓库当前状态

Codex 开始修改前先执行：

```bash
cd ~/SNN

git status
git rev-parse HEAD

git -C fast-hadamard-transform status
git -C fast-hadamard-transform rev-parse HEAD

python - <<'PY'
import fast_hadamard_transform
print(fast_hadamard_transform.__file__)
from fast_hadamard_transform import hadamard_transform
print("fast-hadamard-transform import OK")
PY
```

必须确认 Python 输出路径指向：

```text
/home/wangwenkang/SNN/fast-hadamard-transform/fast_hadamard_transform/...
```

如果不是，停止继续修改并明确报错，不要静默改用其他安装版本。

同时检查：

```bash
test -f .gitmodules && cat .gitmodules || true
git ls-files -s fast-hadamard-transform
```

记录父仓库当前对该目录的 gitlink 状态。

---

# 2. `fast-hadamard-transform` 的目标状态

## 2.1 固定上游仓库与 revision

项目固定使用：

```text
Repository:
https://github.com/Dao-AILab/fast-hadamard-transform.git

Pinned revision:
e7706faf8d1c3b9f241e36860640ad1dac644ede
```

该 revision 必须同时体现在：

1. `.gitmodules` 的 submodule URL；
2. 父仓库记录的 `fast-hadamard-transform` gitlink commit；
3. `环境配置.md` 中的文字说明。

不要使用另一个随机 revision。

---

## 2.2 规范 `.gitmodules`

项目根目录必须存在 `.gitmodules`，内容至少为：

```ini
[submodule "fast-hadamard-transform"]
    path = fast-hadamard-transform
    url = https://github.com/Dao-AILab/fast-hadamard-transform.git
```

要求：

- 使用 HTTPS URL，避免新环境强依赖 GitHub SSH key；
- `path` 必须保持 `fast-hadamard-transform`；
- 不要改目录名；
- 不要 vendor/复制整个 dependency 到 `snn2/`。

如果父仓库当前已经把 `fast-hadamard-transform` 记录为 gitlink，但 `.gitmodules` 缺失，则补齐 `.gitmodules`，不要重新把整个目录当普通文件提交。

---

## 2.3 把 submodule 固定到指定 revision

在服务器执行：

```bash
cd ~/SNN

git -C fast-hadamard-transform fetch origin
git -C fast-hadamard-transform checkout e7706faf8d1c3b9f241e36860640ad1dac644ede
```

然后确认：

```bash
git -C fast-hadamard-transform rev-parse HEAD
```

必须输出：

```text
e7706faf8d1c3b9f241e36860640ad1dac644ede
```

父仓库随后应该把该 gitlink 更新到此 revision。

不要在 submodule 内产生本项目自己的额外 commit。

---

## 2.4 editable install 是正式安装方式

在 `环境配置.md` 中，Fast Hadamard 部分改成以 **submodule + editable install** 为主流程。

推荐的新环境流程必须写成：

```bash
git clone --recurse-submodules https://github.com/wangwk699/SNN.git
cd SNN

git submodule update --init --recursive

python -m pip install -e ./fast-hadamard-transform --no-build-isolation
```

如果仓库已经 clone 但 submodule 尚未初始化：

```bash
git submodule update --init --recursive
python -m pip install -e ./fast-hadamard-transform --no-build-isolation
```

随后必须验证：

```bash
python - <<'PY'
import fast_hadamard_transform
print(fast_hadamard_transform.__file__)
from fast_hadamard_transform import hadamard_transform
print("fast-hadamard-transform OK")
PY
```

文档要明确：

- `-e` 是 editable install；
- Python 会直接使用当前项目 `fast-hadamard-transform/` 目录对应的源码/扩展；
- 仅仅目录存在并不足以保证 CUDA extension 可用，仍需执行 editable install；
- 正式 CUDA Rotation 不允许在 Fast Hadamard 失败后静默退回纯 Torch。

---

# 3. `_fast_fht()` 的最终要求

用户已经修改过该函数。Codex 只需要检查其最终语义符合以下要求，若已经满足，不要重复重构。

文件：

```text
snn2/hadamard.py
```

目标语义：

```python
def _fast_fht(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return _pure_fht(x, normalized=True)

    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError as exc:
        raise RuntimeError(
            "CUDA Hadamard requires fast-hadamard-transform. "
            "Install the local package with: "
            "python -m pip install -e ./fast-hadamard-transform --no-build-isolation"
        ) from exc

    try:
        return hadamard_transform(
            x.contiguous(),
            scale=1.0 / math.sqrt(x.shape[-1]),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "fast-hadamard-transform failed on CUDA; "
            "refusing to silently fall back to the PyTorch implementation."
        ) from exc
```

允许文字稍有不同，但必须满足：

```text
CPU:
    pure Torch fallback allowed

CUDA:
    fast-hadamard-transform required
    ImportError -> hard failure
    RuntimeError -> hard failure
    no silent fallback
```

不要引入新的 backend selector 配置。

---

# 4. Rotation regression check：设计要求

## 4.1 核心目的

当前 `prepare_rotation.py` 会把原始 Base 权重通过 fixed random Hadamard rotation 重参数化。

Rotation 理论上应保持模型函数基本不变：

\[
f_{\mathrm{Base}}(x)
\approx
f_{\mathrm{Rotated}}(x).
\]

因为实际运行包含 BF16/FP32 转换、Hadamard 运算以及重新保存权重，所以允许小的数值误差，但不能出现显著 logits 偏离。

因此必须新增**模型级 regression check**。

---

## 4.2 必须使用已有 calibration dataset

不要重新随机抽数据。

必须读取项目已经生成的：

```text
artifacts/<experiment>/<task>/_shared/seed42/data/calibration_manifest.json
```

并通过现有：

```python
load_selected_raw(cfg, layout).calibration
```

取得固定的 128 个 calibration samples。

原因：

```text
Vanilla analysis
ANN-training calibration
Post-finetuning calibration
Rotation regression
```

都应基于同一套 calibration sample selection。

不要创建新的 manifest。

不要改变：

```text
calibration.num_samples = 128
calibration.seed = 42
sampling = without replacement
```

---

## 4.3 Regression check 不允许使用 Prefix

Regression 的比较双方必须是：

```text
Original Base
vs
Rotated Base
```

必须：

```text
no Prefix
no fixed Prefix KV
no prefix token concatenation
```

原因是该测试只验证 Rotation 本身是否保持函数。

即使当前 `ann_training` 后续会使用 Prefix，这一步也不能加入 Prefix。

---

## 4.4 Regression check 不允许 activation replacement

Rotated model 必须安装完整 SNN2 Rotation integration，但 controller 使用：

```python
SiteController(mode="identity")
```

`identity` 模式下 Site 1–10 都原样返回 activation，不得加载 Phase/GIF/MTN/Clip states。

不要使用：

```text
phase
gif
deploy_phase
deploy_gif
deploy_mtn
collect
```

作为该 regression 的 controller mode。

---

## 4.5 Rotated model 必须包含 online R3 / R4

这是本检查最重要的实现要求之一。

当前 rotation architecture 是：

```text
offline / fused:
    R1
    R2
    R4 inverse fused into down_proj side

online:
    R3 after RoPE for Q/K
    R4 before down_proj
```

因此 regression 中不能只执行：

```python
fuse_rotations(rotated_model)
```

然后直接 forward。

必须在 fused model 上安装：

```python
install_model_integration(
    rotated_model,
    SiteController(mode="identity"),
    rotation_state,
)
```

从而实际 forward 包含：

```text
online R3
online R4
```

否则比较对象不是项目真正的 Rotated Base。

---

# 5. 建议的代码组织

优先保持代码职责清晰。

建议在 `snn2/rotation.py` 中增加一个独立函数，例如：

```python
validate_rotation_logits(...)
```

或：

```python
rotation_regression_check(...)
```

不要把几十行比较逻辑全部堆进 `scripts/prepare_rotation.py`。

该 helper 应负责：

1. 接收：
   - original Base model；
   - fully integrated rotated model；
   - tokenizer；
   - calibration dataset；
   - cfg；
   - device / batch policy；
2. 对 128 samples 做 forward；
3. 聚合误差统计；
4. 返回一个 JSON-serializable dict；
5. 如果违反 acceptance criterion，则抛出明确异常。

`prepare_rotation.py` 负责：

```text
load models
fuse rotations
install identity integration
load calibration samples
call regression helper
write regression JSON
only then save successful rotation artifacts
```

如果实际代码结构决定把 helper 放入单独模块，例如：

```text
snn2/rotation_validation.py
```

也可以，但不要引入不必要复杂度。

---

# 6. Base / Rotated 模型构造

推荐方案：

```python
base_model = load_model(
    cfg,
    cfg["experiment"]["model_name"],
    training=False,
    ...
)

rotated_model = load_model(
    cfg,
    cfg["experiment"]["model_name"],
    training=False,
    ...
)

rotation_state = fuse_rotations(
    rotated_model,
    seed=int(cfg["rotation"]["seed"]),
    device=cfg["rotation"].get("fusion_device", "cuda"),
)

controller = SiteController(mode="identity")

install_model_integration(
    rotated_model,
    controller,
    rotation_state,
)
```

注意：

- 两个模型必须来自同一个原始 pretrained revision；
- 两个模型均：
  ```python
  model.eval()
  ```
- comparison 必须在：
  ```python
  torch.inference_mode()
  ```
  下进行；
- forward 统一：
  ```python
  use_cache=False
  ```
- 不使用 Prefix。
- tokenizer 必须相同。

如果同时持有两个 8B 模型在目标服务器上会造成 OOM，可以采用**内存安全但仍严格等价**的实现，例如：
- 先逐 sample 或逐小 batch 计算 Base logits 的必要比较摘要/临时张量；
- 或合理地把其中一个模型放 CPU；
- 或按样本计算后立即释放；
但不能因为显存问题删掉 regression check，也不能只检查一个 toy sample。

优先基于当前服务器 A100 80GB 环境选择最简单可靠实现。

---

# 7. Tokenization 与 batch

必须使用项目现有 tokenization semantics，不要另外写一套 prompt 编码逻辑。

优先复用：

```python
tokenize_dataset(...)
CausalLMCollator(...)
```

或项目当前 calibration 统计过程中已经使用的等价 tokenization helper。

要求：

- 128 个 calibration samples 全部覆盖；
- `prefix_ids=None`；
- 使用 `cfg["data"]["max_seq_length"]` 等当前配置；
- 不改变 truncation；
- 不改变 special token 逻辑；
- batch size 可以为 1，优先保证确定性和显存安全。

---

# 8. Logits 比较指标

至少输出：

```text
num_samples
num_tokens_compared
max_abs_error
mean_abs_error
relative_l2_error
passed
```

定义：

\[
\text{max\_abs\_error}
=
\max |\ell_b-\ell_r|
\]

\[
\text{mean\_abs\_error}
=
\mathrm{mean}|\ell_b-\ell_r|
\]

\[
\text{relative\_l2\_error}
=
\frac{\|\ell_b-\ell_r\|_2}
{\|\ell_b\|_2+\epsilon}.
\]

其中：

```text
ell_b = Base logits
ell_r = Rotated logits
```

聚合必须覆盖所有 128 calibration samples 的有效 forward 输出。

可以额外输出：

```text
base_logits_l2
rotated_logits_l2
dtype
model_name
rotation_seed
calibration_manifest_path
calibration_manifest_sha256
```

但这不是必须项。

---

# 9. Acceptance criterion

不要把模型等价性写成 bitwise equality。

BF16 / CUDA / FHT 允许有合理浮点误差。

实现时优先使用项目中已有数值测试经验或先运行 Qwen3-1.7B 得到自然误差，再选择一个统一的、保守且能捕获 Rotation 实现错误的 threshold。

必须满足两个原则：

1. threshold 要写入代码或 config，不能只人工查看；
2. failure 必须：
   ```python
   raise RuntimeError(...)
   ```
   阻止错误的 `fused_base` 继续进入实验。

如果 Codex 无法从项目现有测试确定可靠阈值，则：
- 不要拍脑袋使用极端严格的 `1e-5`；
- 可采用一个明确、保守的相对误差准则，并在结果 JSON 中同时记录完整三个误差指标；
- 在最终修改说明中明确给出所选 threshold 和理由。

---

# 10. Regression 输出位置

建议保存：

```text
artifacts/<experiment>/<task>/<model>/_shared/seed42/rotated_prefix/rotation/
├── rotation_state.pt
├── rotation_summary.json
├── rotation_regression.json
└── fused_base/
```

文件名固定建议：

```text
rotation_regression.json
```

JSON 示例：

```json
{
  "format_version": 1,
  "purpose": "base_vs_rotated_logits_regression",
  "model_name": "Qwen/Qwen3-1.7B-Base",
  "rotation_seed": 42,
  "num_samples": 128,
  "num_tokens_compared": 12345,
  "max_abs_error": 0.03125,
  "mean_abs_error": 0.0008,
  "relative_l2_error": 0.0004,
  "threshold": {
    "relative_l2_error": 0.001
  },
  "passed": true
}
```

数字只是结构示例，不要直接把示例数值作为实际阈值或结果。

---

# 11. `prepare_rotation.py` 的失败语义

目标流程应为：

```text
load Original Base A
load Original Base B
        ↓
对 B 执行 fuse_rotations()
        ↓
给 B 安装 identity controller + online R3/R4
        ↓
读取已有 128 calibration samples
        ↓
Base A vs Rotated B logits regression
        ↓
PASS
        ↓
保存/确认 rotation regression result
        ↓
保存 rotation_state
        ↓
保存 fused_base
```

如果 regression FAIL：

```text
必须抛异常
不得把此次 rotation 当作成功的 shared preprocessing
不得继续 Prefix discovery / calibration / ANN training
```

如果为了内存安全需要先保存临时 fused model 再比较，可以调整具体保存顺序，但最终成功工件必须明确记录 regression passed；失败不能留下一个看起来可继续使用的成功状态。

---

# 12. 不要增加的测试

用户明确要求：

```text
暂时不做不同 seed randomized rotation 的
absolute-value distribution 差异检查。
```

因此不要新增类似：

```python
seed42_abs != seed43_abs
```

的测试。

现有数学级：

```text
QQ^T = I
random_hadamard round trip
```

测试保留。

本轮新增的 model-level regression 只验证：

```text
Base logits ≈ Rotated logits
```

---

# 13. `环境配置.md` 的具体更新要求

当前 `环境配置.md` 中“安装 Dao-AILab Fast Hadamard Transform”部分要重写为规范的 submodule 使用方式。

至少覆盖以下内容：

### 13.1 新 clone

```bash
git clone --recurse-submodules https://github.com/wangwk699/SNN.git
cd SNN

git submodule update --init --recursive

python -m pip install -e ./fast-hadamard-transform --no-build-isolation
```

### 13.2 已有 clone

```bash
cd ~/SNN
git submodule update --init --recursive

python -m pip install -e ./fast-hadamard-transform --no-build-isolation
```

### 13.3 import 来源验证

```bash
python - <<'PY'
import fast_hadamard_transform
print(fast_hadamard_transform.__file__)
from fast_hadamard_transform import hadamard_transform
print("fast-hadamard-transform OK")
PY
```

说明预期路径应落在：

```text
<SNN_ROOT>/fast-hadamard-transform/fast_hadamard_transform/
```

### 13.4 固定 revision

明确写：

```text
Dao-AILab/fast-hadamard-transform
e7706faf8d1c3b9f241e36860640ad1dac644ede
```

### 13.5 正式 CUDA 行为

明确写：

```text
CPU correctness/test path:
    pure Torch implementation allowed

CUDA formal experiment:
    fast-hadamard-transform is mandatory
    import/kernel failure causes hard error
    no silent PyTorch fallback
```

### 13.6 环境自检

把原来的：

```bash
python -c "from fast_hadamard_transform import hadamard_transform; print('fast-hadamard-transform OK')"
```

增强为至少包含 `__file__` 的检查，确保不是误用了另一个 site-packages 版本。

---

# 14. `实验执行总结.md` 的具体更新要求

不要重写整个实验执行文档，只在合适位置补充 Rotation regression。

## 14.1 总体流程

当前 Step 3：

```text
Rotation / fused Base
ANN-training Prefix
ANN-training calibration
Vanilla analysis calibration
```

改成表达：

```text
Rotation / fused Base
└── 自动进行 Base ↔ Rotated logits regression（固定 128 calibration samples）
ANN-training Prefix
ANN-training calibration
Vanilla analysis calibration
```

不要新增一条需要用户手工单独运行的命令，优先让它成为：

```bash
python scripts/prepare_rotation.py --config "$ROT_CFG"
```

内部自动执行的检查。

---

## 14.2 `prepare_rotation.py` 说明

在 `7.4 四条命令分别做什么` 中，`prepare_rotation.py` 的作用应改成至少包含：

```text
1. 加载原始 pretrained Base；
2. 执行 RMSNorm scale fusion 和 R1/R2/R4-inverse 等 offline/fused rotation；
3. 安装 online R3/R4，controller=identity；
4. 读取已经固定的 calibration manifest 对应 128 个 samples；
5. 在 no Prefix、no replacement、use_cache=False 条件下，
   比较 Original Base 与完整 Rotated Base logits；
6. regression 通过后保存 rotation_state / rotation_summary /
   rotation_regression / fused_base；
7. regression 未通过则直接报错并停止，不允许继续后续 Prefix /
   calibration / training。
```

---

## 14.3 输出目录说明

更新为：

```text
_shared/seed42/rotated_prefix/rotation/
├── rotation_state.pt
├── rotation_summary.json
├── rotation_regression.json
└── fused_base/
```

---

## 14.4 用户执行命令保持不变

例如 Qwen3-1.7B：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml
VAN_CFG=configs/generated/exp1_qwen3_1_7b_tldr__vanilla.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"

python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$ROT_CFG" \
  --stage ann_training

python scripts/calibrate_sites.py \
  --config "$VAN_CFG" \
  --stage vanilla_analysis
```

但要在命令下方明确写：

> `prepare_rotation.py` 只有在 `rotation_regression.json` 中 `passed=true` 后才视为成功。若 regression 失败，立即停止该 model-task pair 的后续实验。

Qwen3-8B 与 Llama3-8B 对应章节同样适用，不必复制大量重复解释；可以在共有说明处统一声明。

---

# 15. Tests

## 15.1 保留已有 Hadamard tests

例如：

```text
tests/test_hadamard.py
```

现有：

```text
Paley orthogonality
random_hadamard round trip
```

保留。

---

## 15.2 新增轻量单元测试

不要在普通 `pytest -q` 中强制下载/加载真实 1.7B/8B 模型。

因此真实的 128-sample Base ↔ Rotated regression 应属于 `prepare_rotation.py` 的 runtime validation，而不是普通 CI 单元测试。

但应为 regression helper 增加轻量测试，使用 tiny/mock model 或 synthetic logits，验证：

```text
1. metric aggregation 正确；
2. passed/fail threshold 逻辑正确；
3. fail 时会抛 RuntimeError；
4. JSON 内容可序列化；
5. 确认 regression path 不启用 Prefix；
6. controller 使用 identity。
```

如果为了保持实现简单只能覆盖前 3–4 项，也至少保证 metric 与 hard-fail 逻辑有单元测试。

---

# 16. provenance / artifact verification

检查当前 `scripts/verify_artifacts.py`。

如果它目前负责验证 shared Rotation 工件，则将：

```text
rotation_regression.json
```

加入 rotated modes 的 required artifacts，并至少验证：

```text
purpose == "base_vs_rotated_logits_regression"
num_samples == 128
passed == true
```

若 regression JSON 中记录了 calibration manifest path/SHA，则进一步验证其对应当前 task-level：

```text
calibration_manifest.json
```

并且 hash 一致。

优先建议纳入验证，因为 `verify_artifacts.py` 本来就是最终 artifact provenance gate。

---

# 17. Rotation summary 建议

在 `rotation_summary.json` 中可以增加：

```text
hadamard_backend = "fast_hadamard_transform"
```

以及：

```text
rotation_regression_path = ...
```

若容易实现，也可以记录实际 package source：

```text
fast_hadamard_transform_module_path
```

但不要依赖 Python package 有稳定 `__version__` 字段。

最重要的是：

```text
CUDA 正式运行时不能 silent fallback
rotation_regression passed
```

---

# 18. 必须保持不变的项目协议

本轮不得改变：

```text
3 model-task pairs
4 ANN modes
12 final ANN checkpoints

10 activation replacement sites

Vanilla analysis calibration
ANN-training calibration
Post-finetuning conversion calibration
三者严格分离

Prefix 采用 fixed past_key_values
不把 Prefix token prepend 到 input_ids

Vanilla ANN training:
    no rotation
    no prefix
    no replacement

Rotated modes:
    fused Base + online rotation

Post-finetuning:
    每个 final ANN checkpoint 独立 rediscover Prefix
    独立 conversion calibration

TL;DR / lm-eval Base baseline 语义
Learning-rate run path
final checkpoint path
```

不要因为新增 regression check 改变上述逻辑。

---

# 19. 修改完成后的检查命令

完成代码与文档修改后执行：

```bash
cd ~/SNN

python -m compileall -q snn2 scripts tests
python -m pytest -q
```

检查 submodule：

```bash
cat .gitmodules
git submodule status
git -C fast-hadamard-transform rev-parse HEAD
```

必须看到 pinned revision：

```text
e7706faf8d1c3b9f241e36860640ad1dac644ede
```

检查 Python 实际 import：

```bash
python - <<'PY'
import fast_hadamard_transform
print(fast_hadamard_transform.__file__)
from fast_hadamard_transform import hadamard_transform
print("fast-hadamard-transform OK")
PY
```

必须来自当前 SNN 子目录。

---

# 20. 最小真实实验验收：Qwen3-1.7B TL;DR

确保已经有 task-level data manifest 后：

```bash
ROT_CFG=configs/generated/exp1_qwen3_1_7b_tldr__unaware.yaml

python scripts/prepare_rotation.py \
  --config "$ROT_CFG"
```

成功后必须存在：

```text
artifacts/snn2_main_v1/tldr/Qwen_Qwen3-1.7B-Base/_shared/seed42/rotated_prefix/rotation/
├── rotation_state.pt
├── rotation_summary.json
├── rotation_regression.json
└── fused_base/
```

检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/snn2_main_v1/tldr/"
    "Qwen_Qwen3-1.7B-Base/_shared/seed42/"
    "rotated_prefix/rotation/rotation_regression.json"
)

data = json.loads(path.read_text())
print(json.dumps(data, indent=2))

assert data["purpose"] == "base_vs_rotated_logits_regression"
assert data["num_samples"] == 128
assert data["passed"] is True
PY
```

只有此检查通过后才继续：

```bash
python scripts/discover_prefix.py \
  --config "$ROT_CFG" \
  --stage ann_training
```

然后再继续 ANN-training calibration。

---

# 21. 代码实现时需要特别警惕的问题

### 21.1 不要比较“只有 fused weights、没有 R3/R4”的模型

错误：

```text
Base
vs
fuse_rotations(model) 后直接 forward
```

正确：

```text
Base
vs
fuse_rotations(model)
+ install_model_integration(identity, rotation_state)
后 forward
```

### 21.2 不要把 Prefix 混入 regression

该检查不是：

```text
Base
vs
Rotation + Prefix
```

而是：

```text
Base
vs
Rotation
```

### 21.3 不要启用 collect controller

必须是：

```text
identity
```

### 21.4 不要新增随机 seed distribution test

本轮明确不做。

### 21.5 不要重新采样 calibration data

必须直接使用已有 manifest 指定的固定 128 samples。

### 21.6 不要把 `fast-hadamard-transform` 失败吞掉

CUDA 上失败必须报错。

### 21.7 不要破坏 submodule 工作树

只把 submodule checkout 到指定 upstream revision，不要在其内部修改第三方源代码。

---

# 22. 最终 Codex 交付说明要求

修改完成后，Codex 最终回复必须简洁列出：

1. 修改了哪些文件；
2. `.gitmodules` 最终内容；
3. `fast-hadamard-transform` pinned revision；
4. `_fast_fht()` 是否确认 CUDA hard dependency；
5. Rotation regression 的实现位置；
6. regression 比较是否：
   ```text
   128 calibration samples
   no Prefix
   no replacement
   identity controller
   online R3/R4 included
   ```
7. `rotation_regression.json` 的保存位置；
8. 采用的 acceptance threshold；
9. `pytest -q` 结果；
10. 若实际运行了 Qwen3-1.7B `prepare_rotation.py`，报告实际：
    ```text
    max_abs_error
    mean_abs_error
    relative_l2_error
    passed
    ```
11. `git status`，并明确是否存在未预期修改。

不要只回复“已完成”。

---

# 23. 本轮修改完成标准

只有同时满足以下条件才算完成：

```text
[ ] .gitmodules 存在且 URL 正确
[ ] 父仓库 gitlink 固定到 e7706faf...
[ ] 环境配置.md 使用 submodule + editable install 流程
[ ] 环境配置.md 明确检查 fast_hadamard_transform.__file__
[ ] CUDA _fast_fht 无 silent fallback
[ ] prepare_rotation.py 自动执行 128-sample Base ↔ Rotated logits regression
[ ] regression 使用同一 calibration manifest
[ ] regression 不使用 Prefix
[ ] regression 不使用 activation replacement
[ ] regression 包含 online R3 / R4
[ ] regression failure 为 hard failure
[ ] rotation_regression.json 被保存
[ ] 实验执行总结.md 已更新执行说明
[ ] 不增加不同 seed absolute-value distribution test
[ ] 单元测试通过
[ ] 项目既有实验协议未被改变
```
