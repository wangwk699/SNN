# HISTORICAL MIGRATION PLAN — COMPLETED

> 当前代码已经是 10-site；本文件仅保留完成的 9-site → 10-site 迁移背景，不描述当前实现。

# SNN 项目：Activation Replacement 从 9 个 Site 扩展到 10 个 Site 的完整实施说明

> **目标读者**：部署在服务器上的 Codex / 代码修改智能体  
> **目标仓库**：/home/wangwenkang/SNN  
> **任务性质**：对现有 SNN 实验代码做一次拓扑级修改，将每个 Transformer Block 内的 activation replacement site 从 **9 个扩展到 10 个**，并同步更新 calibration、conversion、evaluation、artifact validation、测试、配置和仓库内 Markdown 文档。  
> **重要要求**：本文件应被视为该任务的完整上下文。执行者不应依赖其它对话信息。

---

## 0. 执行前要求

开始修改前：

1. 进入 SNN 仓库根目录。
3. 阅读以下当前实现文件后再修改：
   - `snn2/model_integration.py`
   - `snn2/stats.py`
   - `snn2/controller.py`
   - `snn2/calibration.py`
   - `snn2/conversion.py`
   - `snn2/config.py`
   - `scripts/evaluate_tldr.py`
   - `scripts/evaluate_lm_harness.py`
   - `scripts/verify_artifacts.py`
   - `configs/experiment_matrix.yaml`
   - `tests/`
   - 仓库中所有 `*.md`
4. 修改前使用 `rg` 全仓库搜索所有与“9 个 site”有关的硬编码和文字描述，至少执行类似：
   ```bash
   rg -n \
     '九个|9 sites|9 site|nine sites|Expected nine|expected_sites_per_layer|× 9|\* 9|site_09_post_mlp_product_r4' \
     .
   ```
5. **不要只机械地把数字 `9` 全局替换为 `10`。**  
   很多 `9` 可能与 site 数量无关。必须逐处理解语义后修改。

本任务只处理 **activation replacement 9-site → 10-site** 及其直接派生影响。不要顺手重构或修复与本任务无关的训练 checkpoint、Prefix、数据、DeepSpeed 等问题。

---

# 1. 任务目标与新的网络拓扑

当前项目每层 Transformer Block 有 9 个 activation replacement site。

本次新增一个 site：

> **新增位置位于 MLP 的 `up_proj` 输出之后、gate-up elementwise product 之前。**

也就是：

```text
x
│
RMSNorm
│
Site 7
├──────────────────────────────┐
│                              │
R1^-1 W_gate                 R1^-1 W_up
│                              │
SiLU                           │
│                              │
Site 8                         Site 9   ← 新增
│                              │
└────────────── ⊙ ─────────────┘
                 │
                R4
                 │
               Site 10         ← 原 Site 9 顺延
                 │
          R4^-1 W_down R1
                 │
               output
```

新增 Site 9 的输入 **不经过额外 rotation**。

它与以下已有位置相同，属于 Identity / `I` 坐标：

- Softmax 后的 Site 5；
- SiLU 后的 Site 8；
- 新增的 `up_proj` 后 Site 9。

**不新增 R5，不修改 R1/R2/R3/R4 的数学定义或 rotation fusion。**

---

# 2. 新的 10-site 唯一定义

修改后，项目中每层必须严格包含以下 10 个 site。

| Site | 名称 | 位置 | 坐标 / Rotation |
|---:|---|---|---|
| 1 | `post_input_rmsnorm` | Self-Attention 前 RMSNorm 后 | `R1` |
| 2 | `q_post_rope_r3` | Q 经 RoPE 与在线 R3 后 | `R3` |
| 3 | `k_post_rope_r3` | K 经 RoPE 与在线 R3 后 | `R3` |
| 4 | `v_projection_r2` | V projection 的 R2 坐标输出 | `R2` |
| 5 | `post_spiking_softmax` | Softmax 后 | `I` |
| 6 | `post_attention_value_dot_r2` | Attention weight × V 后 | `R2` |
| 7 | `post_mlp_rmsnorm` | MLP 前 RMSNorm 后 | `R1` |
| 8 | `post_spiking_silu` | `gate_proj → SiLU` 后 | `I` |
| 9 | `post_mlp_up_proj` | `up_proj` 后、elementwise product 前 | `I` |
| 10 | `post_mlp_product_r4` | `gate ⊙ up → R4` 后 | `R4` |

统一映射：

```text
Site:      1   2   3   4   5   6   7   8   9   10
Rotation: R1  R3  R3  R2   I  R2  R1   I   I   R4
```

其中：

- Site 1–8 的语义保持不变；
- **新增 Site 9 = `up_proj` 输出之后；**
- **旧 Site 9 = `post_mlp_product_r4`，必须整体重编号为 Site 10。**

---

# 3. 建立统一的 Site Topology 单一事实来源

## 3.1 新增 `snn2/sites.py`

当前项目把 site 名称和 site 数量散落在多个模块中，存在硬编码 `9`。

本次必须新增：

```text
snn2/sites.py
```

作为全项目 site topology 的唯一事实来源。

至少定义：

```python
SITE_TOPOLOGY_VERSION = 2

SITE_NAMES = {
    1: "post_input_rmsnorm",
    2: "q_post_rope_r3",
    3: "k_post_rope_r3",
    4: "v_projection_r2",
    5: "post_spiking_softmax",
    6: "post_attention_value_dot_r2",
    7: "post_mlp_rmsnorm",
    8: "post_spiking_silu",
    9: "post_mlp_up_proj",
    10: "post_mlp_product_r4",
}

SITE_COORDINATES = {
    1: "R1",
    2: "R3",
    3: "R3",
    4: "R2",
    5: "I",
    6: "R2",
    7: "R1",
    8: "I",
    9: "I",
    10: "R4",
}

SITE_IDS = tuple(sorted(SITE_NAMES))
SITE_COUNT = len(SITE_IDS)
```

并把原先 `stats.py` 中的：

```python
def site_key(layer_index: int, site_index: int) -> str:
    ...
```

迁移到这里。

建议定义为：

```python
def site_key(layer_index: int, site_index: int) -> str:
    if site_index not in SITE_NAMES:
        raise ValueError(f"Unknown activation replacement site: {site_index}")
    return (
        f"layer_{layer_index:03d}/"
        f"site_{site_index:02d}_{SITE_NAMES[site_index]}"
    )
```

同时提供便于 exact-topology validation 的 helper，例如：

```python
def expected_site_dirnames() -> set[str]:
    return {
        f"site_{site_index:02d}_{SITE_NAMES[site_index]}"
        for site_index in SITE_IDS
    }
```

可以再提供一个 metadata helper：

```python
def topology_metadata() -> dict:
    return {
        "site_topology_version": SITE_TOPOLOGY_VERSION,
        "site_count": SITE_COUNT,
        "site_names": {str(k): v for k, v in SITE_NAMES.items()},
        "site_coordinates": {str(k): v for k, v in SITE_COORDINATES.items()},
    }
```

要求：

- 项目其它模块不要再各自定义 site 数量；
- 不要在新代码中继续硬编码 `10` 作为计算逻辑；
- 应使用 `SITE_COUNT`；
- 与 artifact 相关的 metadata 尽量写入 `SITE_TOPOLOGY_VERSION`、`SITE_COUNT`、site names 和 coordinates。

## 3.2 `snn2/stats.py`

将原本定义在 `stats.py` 内的 `SITE_NAMES` / `site_key` 改为从 `snn2.sites` 导入。

为了减少潜在的外部 import 破坏，可以在 `stats.py` 中保留 re-export，例如：

```python
from .sites import (
    SITE_COORDINATES,
    SITE_COUNT,
    SITE_IDS,
    SITE_NAMES,
    SITE_TOPOLOGY_VERSION,
    site_key,
    topology_metadata,
)
```

但 `SITE_NAMES` 的真正定义只能有一份，即 `snn2/sites.py`。

---

# 4. 修改 MLP forward：真正加入新 Site 9

核心文件：

```text
snn2/model_integration.py
```

当前 `_make_mlp_forward(...)` 的逻辑大致是：

```python
gate = mlp.act_fn(mlp.gate_proj(x))
gate = controller.apply(layer_index, 8, gate)

up = mlp.up_proj(x)

if controller.mode == "collect":
    controller.record_saliency(
        layer_index,
        8,
        gate.square() * up.square(),
    )

product = gate * up

if r4 is not None:
    product = random_hadamard(product, r4)

product = controller.apply(layer_index, 9, product)

return mlp.down_proj(product)
```

必须修改为以下语义：

```python
gate = mlp.act_fn(mlp.gate_proj(x))
gate = controller.apply(layer_index, 8, gate)

up = mlp.up_proj(x)
up = controller.apply(layer_index, 9, up)

if controller.mode == "collect":
    product_saliency = gate.square() * up.square()
    controller.record_saliency(layer_index, 8, product_saliency)
    controller.record_saliency(layer_index, 9, product_saliency)

product = gate * up

if r4 is not None:
    product = random_hadamard(product, r4)

product = controller.apply(layer_index, 10, product)

return mlp.down_proj(product)
```

必须满足以下顺序：

```text
gate_proj
   ↓
 SiLU
   ↓
Site 8 ──────────────┐
                     │
                     ⊙
                     │
up_proj              │
   ↓                 │
Site 9 ──────────────┘
        ↓
       R4
        ↓
     Site 10
        ↓
    down_proj
```

## 4.1 新 Site 9 不经过 rotation

不要在：

```text
up_proj → Site 9
```

之间插入任何 Hadamard。

新增 Site 9 的 coordinate 必须是：

```text
I
```

`R4` 仍然只能位于：

```text
Site 8 与 Site 9 的输出做 product
        ↓
       R4
        ↓
     Site 10
```

---

# 5. GIF operator-aware saliency：Site 8 / Site 9 对称处理

当前 Site 8 是 gate-up elementwise product 的一个操作数。

定义：

```text
g = Site 8 的 activation
u = Site 9 的 activation
y = g ⊙ u
```

对于 elementwise product：

```text
y = g ⊙ u
```

本项目继续使用当前的 product-aware sensitivity：

```python
g.square() * u.square()
```

新增 Site 9 后，必须分别给 Site 8 和 Site 9 保存 operator-aware saliency。

实现：

```python
if controller.mode == "collect":
    product_saliency = gate.square() * up.square()
    controller.record_saliency(layer_index, 8, product_saliency)
    controller.record_saliency(layer_index, 9, product_saliency)
```

也就是说：

```text
Site 8 saliency = g² ⊙ u²
Site 9 saliency = u² ⊙ g²
```

两者数值相同，但必须记录到各自独立的 calibration site。

原因：

- Site 8 是 elementwise product operand 1；
- Site 9 是 elementwise product operand 2；
- 两个 activation 都会独立建立 Phase / GIF / MTN / clip state。

---

# 6. 原 Site 9 的 down_proj saliency 必须迁移到 Site 10

`install_model_integration(...)` 中目前 `down_input_hook(...)` 为 MLP product / `down_proj` consumer 记录线性敏感度。

旧实现类似：

```python
def down_input_hook(_module, inputs, output, index=layer_index):
    controller.record_saliency(
        index,
        9,
        _linear_score(inputs[0], output, _module.weight),
    )
```

必须改成：

```python
def down_input_hook(_module, inputs, output, index=layer_index):
    controller.record_saliency(
        index,
        10,
        _linear_score(inputs[0], output, _module.weight),
    )
```

注意：

- Site 9 的 activation statistics 必须对应 `up_proj` 输出；
- Site 9 的 saliency 也必须对应这个 `up_proj` branch 作为 elementwise-product operand 的 sensitivity；
- Site 10 才对应 R4 后 product → `down_proj` 的 linear-consumer sensitivity。

绝不能出现：

```text
Site 9 activation = up_proj 输出
Site 9 saliency   = down_proj 输入 sensitivity
```

这样的错位。

---

# 7. SiteController / Neuron 不需要新的算法分支

核心文件：

```text
snn2/controller.py
snn2/neurons.py
```

原则上不需要为 Site 9 新写 Phase/GIF/MTN neuron。

原因：

`SiteController.apply(layer_index, site_index, x)` 已经通过：

```text
layer_xxx/site_xx_name/
```

动态加载：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

因此只要新 Site 9 calibration states 正确生成，controller 即可通用处理。

要求：

- 检查 `controller.py` 是否仍从 `stats.py` 导入 `site_key`；
- 推荐改成从 `snn2.sites` 导入；
- 确保 controller 对 site 10 没有任何隐式范围限制；
- `neurons.py` 不要因为本任务进行不必要算法修改。

---

# 8. Calibration 从 9-site 升级为 10-site

核心文件：

```text
snn2/calibration.py
configs/experiment_matrix.yaml
snn2/config.py
```

## 8.1 配置

把：

```yaml
calibration:
  expected_sites_per_layer: 9
```

改成：

```yaml
calibration:
  expected_sites_per_layer: 10
```

然后重新生成全部 resolved run configs：

```bash
python scripts/materialize_configs.py
```

必须确认 `configs/generated/*.yaml` 中全部 12 个 config 都是：

```yaml
expected_sites_per_layer: 10
```

由于 `configs/generated/` 当前是仓库的一部分，生成后的变更应保留在最终 diff 中。

## 8.2 `snn2/config.py`

在 config validation 中增加 topology 一致性约束。

推荐：

```python
from .sites import SITE_COUNT
```

并验证：

```python
expected = int(cfg["calibration"]["expected_sites_per_layer"])
if expected != SITE_COUNT:
    raise ValueError(
        "calibration.expected_sites_per_layer must match "
        f"the code topology: config={expected}, code={SITE_COUNT}"
    )
```

目的：

防止以后 stale generated config 仍写 9，但代码已经是 10。

## 8.3 Calibration state 生成逻辑

`build_site_states(...)` 本身是 site-agnostic：

只要 statistics 中存在：

```text
value_min
value_max
abs_max
saliency_sum
saliency_row_count
...
```

就可以继续统一生成：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt
```

因此：

**不要为新 Site 9 复制一套独立的 neuron calibration 算法。**

新 Site 9 应自动产生：

```text
layer_xxx/
└── site_09_post_mlp_up_proj/
    ├── statistics.pt
    ├── statistics_summary.json
    ├── phase_state.pt
    ├── gif_state.pt
    ├── mtn_state.pt
    ├── clip_state.pt
    └── calibration_summary.json
```

而新的旧 Site 9 顺延目录：

```text
layer_xxx/
└── site_10_post_mlp_product_r4/
    ├── statistics.pt
    ├── statistics_summary.json
    ├── phase_state.pt
    ├── gif_state.pt
    ├── mtn_state.pt
    ├── clip_state.pt
    └── calibration_summary.json
```

---

# 9. Calibration Validation 必须做“精确 topology 校验”，不能只数到 10

当前代码主要通过：

```text
count == expected_sites_per_layer
```

判断 calibration 是否完整。

新版本不能只改成 `count == 10`。

必须验证：

> 每个 layer 的 site directory/name 集合与新的 10-site topology **完全相等**。

每层期望目录集合：

```text
site_01_post_input_rmsnorm
site_02_q_post_rope_r3
site_03_k_post_rope_r3
site_04_v_projection_r2
site_05_post_spiking_softmax
site_06_post_attention_value_dot_r2
site_07_post_mlp_rmsnorm
site_08_post_spiking_silu
site_09_post_mlp_up_proj
site_10_post_mlp_product_r4
```

检查必须捕获以下 legacy / stale 情况：

```text
site_09_post_mlp_product_r4
```

这个是旧 9-site topology 的 Site 9。

如果它与新的：

```text
site_09_post_mlp_up_proj
site_10_post_mlp_product_r4
```

同时存在，必须报错，不能仅因为“数量 >= 10”而继续。

推荐把 exact topology validation 写成共享 helper，供：

- `snn2/calibration.py`
- `snn2/conversion.py`
- `scripts/verify_artifacts.py`

共同使用。

---

# 10. 防止旧 9-site Calibration 污染新 Calibration

旧 artifact 可能包含：

```text
site_09_post_mlp_product_r4/
```

新代码重新 calibration 时会产生：

```text
site_09_post_mlp_up_proj/
site_10_post_mlp_product_r4/
```

如果直接在旧 `site_root` 上写，会产生旧、新目录混合。

因此必须做 fail-fast 保护。

## 推荐行为

在 calibration 真正执行模型 forward 之前检查 `site_root`。

如果已经存在任何 legacy 9-site directory，例如：

```text
site_09_post_mlp_product_r4
```

或者 site topology metadata 与当前 `SITE_TOPOLOGY_VERSION` 不一致：

- 抛出清晰的 `RuntimeError`；
- 明确提示用户删除/移动旧 calibration `sites/` 后重新运行；
- **不要由代码静默删除用户 artifact。**

允许用户显式清空对应 calibration site 目录后重新生成。

不要自动删除旧 artifact。

---

# 11. Statistics / Calibration Manifest 增加 topology metadata

为了让 artifact 可追溯，建议所有新的 site-related manifest 保存 topology metadata。

至少包括：

```json
{
  "site_topology_version": 2,
  "site_count": 10,
  "site_names": {
    "1": "post_input_rmsnorm",
    "2": "q_post_rope_r3",
    "3": "k_post_rope_r3",
    "4": "v_projection_r2",
    "5": "post_spiking_softmax",
    "6": "post_attention_value_dot_r2",
    "7": "post_mlp_rmsnorm",
    "8": "post_spiking_silu",
    "9": "post_mlp_up_proj",
    "10": "post_mlp_product_r4"
  },
  "site_coordinates": {
    "1": "R1",
    "2": "R3",
    "3": "R3",
    "4": "R2",
    "5": "I",
    "6": "R2",
    "7": "R1",
    "8": "I",
    "9": "I",
    "10": "R4"
  }
}
```

至少更新：

- `statistics_manifest.json`
- `calibration_state_manifest.json`
- `conversion_metadata.json`
- `artifact_verification.json`

建议 evaluation metadata 也保存：

```text
site_topology_version
site_count
```

便于以后检查实验结果究竟来自 9-site 还是 10-site 代码。

---

# 12. `snn2/conversion.py` 改为 10-site exact validation

当前 `validate_calibration(...)` 中存在类似：

```python
if count != 9:
    ...
```

以及：

```text
Expected nine sites per layer
```

必须删除这些旧 9-site 假设。

新的 validation 必须：

1. 从 `snn2.sites` 导入 topology；
2. 验证每层的实际 site set 与 `expected_site_dirnames()` 完全一致；
3. 每个 site 必须存在：
   ```text
   statistics.pt
   phase_state.pt
   gif_state.pt
   mtn_state.pt
   clip_state.pt
   ```
4. 验证 clip interval 合法；
5. 如果存在 `calibration_state_manifest.json`，验证其中：
   ```text
   site_topology_version == SITE_TOPOLOGY_VERSION
   site_count == SITE_COUNT
   ```
6. 返回的 validation metadata 应包含 topology version / site count；
7. conversion metadata 中写入 topology metadata。

错误信息不要再出现：

```text
Expected nine sites per layer
```

而应基于 `SITE_COUNT` 动态构造。

---

# 13. Evaluation：所有 `× 9 sites` 改为 `× SITE_COUNT`

涉及：

```text
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

当前项目统计 activation-site temporal operator 等价计数时使用：

```text
temporal sample steps
× Transformer layers
× 9 sites
```

新定义为：

```text
temporal sample steps
× Transformer layers
× SITE_COUNT
```

当前 `SITE_COUNT == 10`。

代码中应：

```python
from snn2.sites import (
    SITE_COUNT,
    SITE_TOPOLOGY_VERSION,
)
```

然后：

```python
activation_site_temporal_operator_calls = (
    temporal_sample_step_forwards
    * layers
    * SITE_COUNT
)

batched_activation_site_temporal_slots = (
    batched_temporal_sample_slots
    * layers
    * SITE_COUNT
)
```

不要重新硬编码：

```python
* 10
```

同时：

- 修改相应注释中的 `9 replacement sites`；
- 修改结果 metadata；
- 建议新增：
  ```python
  "site_count": SITE_COUNT,
  "site_topology_version": SITE_TOPOLOGY_VERSION,
  ```

---

# 14. `scripts/verify_artifacts.py`

必须同步新的 topology。

要求：

1. 不再以旧“每层九个 site”为标准；
2. 使用 `validate_calibration(...)` 的 exact topology 校验；
3. `artifact_verification.json` 中保存：
   ```text
   site_count = 10
   site_topology_version = 2
   ```
4. 如果旧 9-site artifact 被发现，应明确失败；
5. 不要因为目录数量看起来正确而接受错误的 site 名称。

---

# 15. `snn2/stats.py` 的 variable-length Site 5 保持不变

当前只有 Softmax Site 5 需要 variable-length key-position statistics。

类似：

```python
StatisticsStore(
    max_channels_by_site={
        5: int(cfg["data"]["max_seq_length"])
    }
)
```

这一逻辑保持不变。

新增 Site 9：

```text
post_mlp_up_proj
```

最后一维是固定 intermediate hidden dimension，因此：

- 不加入 `max_channels_by_site`；
- 不使用 variable channels；
- 使用普通 fixed-channel statistics。

---

# 16. Rotation 代码不做算法修改

以下文件原则上不需要因为新增 Site 9 修改 rotation 算法：

```text
snn2/rotation.py
snn2/hadamard.py
```

保持：

- R1：residual hidden space；
- R2：value/head coordinate；
- R3：Q/K after RoPE online；
- R4：MLP gate-up product 后 online。

新增 Site 9 是：

```text
up_proj → Site 9
```

其输入不经过额外 rotation。

**禁止新增 R5。**

---

# 17. Prefix / Prefix KV 不做算法修改

以下文件不应因为本任务改变 Prefix 语义：

```text
snn2/prefix.py
snn2/prefix_cache.py
```

当前 Prefix 使用固定 KV cache 注入，不能改回把 prefix token prepend 到 `input_ids`。

新增 Site 9 与 Prefix discovery 的算法没有直接关系。

如果 `discover_prefix.py` 会安装 `model_integration`，在 `controller.mode="identity"` 下新增的：

```python
controller.apply(layer_index, 9, up)
```

必须保持恒等，因此不应影响 prefix discovery 的模型数值。

---

# 18. Training 语义

四种 ANN mode 的语义保持：

```text
vanilla:
    rotation = false
    prefix = false
    train replacement = none

unaware:
    rotation = true
    prefix = true
    train replacement = none

phase_aware:
    rotation = true
    prefix = true
    train replacement = phase

gif_aware:
    rotation = true
    prefix = true
    train replacement = gif
```

加入 Site 9 后：

## `none` / `identity`

```text
up_proj → Site 9 → up
```

必须完全等价于原始 `up_proj` 输出。

## `phase_aware`

新的 MLP 训练语义：

```text
gate_proj → SiLU → Phase/clip Site 8
up_proj          → Phase/clip Site 9
两者 product
→ R4
→ Phase/clip Site 10
→ down_proj
```

## `gif_aware`

新的 MLP 训练语义：

```text
gate_proj → SiLU → static GIF/clip Site 8
up_proj          → static GIF/clip Site 9
两者 product
→ R4
→ static GIF/clip Site 10
→ down_proj
```

---

# 19. Full-temporal SNN Deployment 语义

新增 Site 9 必须和其它 site 一样参与 full-temporal deployment。

也就是说在：

```text
deploy_phase
deploy_gif
deploy_mtn
```

模式下：

- Site 8 将 gate activation temporalize；
- Site 9 将 up activation temporalize；
- gate 与 up 的 temporal tensor 按当前 timestep 对应元素相乘；
- product 经过 R4；
- Site 10 再进行 neuron temporal operation；
- 再进入 `down_proj`。

不要只在 calibration / train mode 加 Site 9 而遗漏 deploy mode。

只要 Site 9 通过统一：

```python
controller.apply(layer_index, 9, up)
```

进入现有 SiteController，它就应该自然覆盖 collect / phase / gif / deploy_*。

---

# 20. Tests：必须新增 / 修改

修改完成后至少覆盖以下测试。

---

## 20.1 Site topology test

新增例如：

```text
tests/test_sites.py
```

验证：

```python
assert SITE_COUNT == 10
assert SITE_IDS == tuple(range(1, 11))
```

并验证：

```text
Site 8  = post_spiking_silu / I
Site 9  = post_mlp_up_proj / I
Site 10 = post_mlp_product_r4 / R4
```

以及 `site_key(...)`：

```text
site_09_post_mlp_up_proj
site_10_post_mlp_product_r4
```

---

## 20.2 MLP placement test

验证调用顺序：

```text
Site 8
Site 9
Site 10
```

并且：

- Site 8 在 gate SiLU 后；
- Site 9 在 up_proj 后；
- Site 10 在 product（及 R4，如启用）后。

测试要能捕获以下错误：

- Site 9 被放到了 product 后；
- Site 9 被放到了 R4 后；
- Site 10 仍错误编号为 9。

---

## 20.3 Identity parity test

当 controller 为：

```text
identity / none
```

时，新增 Site 9 不得改变网络数值。

至少对 MLP 路径验证：

```text
reference output
≈
10-site integrated output
```

如果使用浮点比较：

```python
torch.testing.assert_close(...)
```

---

## 20.4 Collect / saliency test

在 collect 模式下确认每层 MLP 至少产生：

```text
Site 8 statistics + saliency
Site 9 statistics + saliency
Site 10 statistics + saliency
```

并确认：

```text
Site 8 saliency = gate² * up²
Site 9 saliency = gate² * up²
```

Site 10 saliency 来自：

```text
down_proj linear consumer
```

---

## 20.5 Exact topology validation test

构造或模拟 legacy 目录：

```text
site_09_post_mlp_product_r4
```

确保新 validation 拒绝旧 topology。

再构造新目录：

```text
site_09_post_mlp_up_proj
site_10_post_mlp_product_r4
```

确认只有完整的 10-site exact set 才通过。

---

## 20.6 Existing tests

运行：

```bash
pytest -q
```

至少确保当前已有：

```text
tests/test_hadamard.py
tests/test_neurons.py
tests/test_prefix.py
tests/test_statistics.py
```

不回归。

尤其：

- Prefix test 不能被这次修改破坏；
- Softmax Site 5 variable-length statistics test 不能被破坏。

---

# 21. Generated Config Validation

完成代码与 matrix 修改后运行：

```bash
python scripts/materialize_configs.py
```

然后检查：

```bash
rg -n 'expected_sites_per_layer' configs/generated configs/experiment_matrix.yaml
```

所有主实验配置必须统一：

```text
expected_sites_per_layer: 10
```

不允许：

```text
experiment_matrix.yaml = 10
generated yaml = 9
```

这样的 stale config。

---

# 22. 仓库中的 Markdown 文档必须全部同步为 10-site

这是本任务的强制要求，不是可选项。

必须扫描仓库所有 tracked Markdown：

```bash
find . -name '*.md' -type f -print
```

并搜索旧描述：

```bash
rg -n \
  '九个|9 sites|9 site|nine sites|× 9|site 9|Site 9' \
  --glob '*.md' \
  .
```

逐处判断是否与旧 9-site topology 有关并修改。

**禁止只修改代码而保留旧文档。**

---

# 23. `代码结构总结.md` 必须更新的内容

当前该文档明确包含旧 9-site 描述。

至少修改以下内容。

## 23.1 总体数据流

旧：

```text
在 prefix + 实际旋转坐标下收集每层九个 site 的统计量
```

新：

```text
在 prefix + 实际旋转坐标下收集每层十个 site 的统计量
```

---

## 23.2 目录结构注释

旧描述中的：

```text
calibrate_sites.py   # 收集九个 site ...
verify_artifacts.py  # 校验 ... 九个 site ...
model_integration.py # 九个 site ...
```

全部改成十个 site。

如果新增 `snn2/sites.py`，在代码结构树中必须加入，例如：

```text
snn2/sites.py  # 十个 activation replacement site 的统一 topology 定义
```

---

## 23.3 章节标题

旧：

```markdown
## 4. 九个 activation replacement site
```

改为：

```markdown
## 4. 十个 activation replacement site
```

---

## 23.4 Rotation mapping

旧：

```text
1:R1, 2:R3, 3:R3, 4:R2, 5:I, 6:R2, 7:R1, 8:I, 9:R4
```

改为：

```text
1:R1, 2:R3, 3:R3, 4:R2, 5:I, 6:R2, 7:R1, 8:I, 9:I, 10:R4
```

并明确写出：

- Site 8：SiLU 后，无额外 rotation；
- Site 9：`up_proj` 后，无额外 rotation；
- Site 10：gate-up product 经 R4 后。

原文：

```text
Softmax 后与 SiLU 后不增加 rotation
```

改为类似：

```text
Softmax 后、SiLU 后以及 up_proj 后均不增加额外 rotation；
新增 Site 9 位于 up_proj 输出后，原 R4 后 MLP product site 顺延为 Site 10。
```

---

## 23.5 GIF saliency 描述

旧文档中：

```text
site 8 使用 gate-up elementwise-product sensitivity
```

需要扩展为：

```text
site 8 与 site 9 分别对应 gate/up 两个 elementwise-product operand，
二者均使用 gate² × up² 的 product-aware sensitivity；
site 10 使用 R4 后 product → down_proj 的 linear-consumer sensitivity。
```

同时旧：

```text
为覆盖九个 site 所作的明确推广
```

改为：

```text
为覆盖十个 site 所作的明确推广
```

---

# 24. `实验执行总结.md` 必须更新的内容

至少修改：

旧：

```text
次数 × 层数 × 9 sites
```

新：

```text
次数 × 层数 × 10 sites
```

更推荐写成概念：

```text
次数 × 层数 × 每层 10 个 activation replacement sites
```

旧：

```text
每层九个 calibration site
```

改为：

```text
每层十个 calibration site
```

并建议在 calibration 部分增加一句：

```text
每层固定收集 10 个 activation replacement site；
新增 Site 9 位于 MLP up_proj 后，原 R4 后 product site 顺延为 Site 10。
```

如果文档中提到旧 `site_09_post_mlp_product_r4`，必须改成：

```text
site_10_post_mlp_product_r4
```

---

# 25. 其它 Markdown 文档

即便目前已知主要旧描述在：

```text
代码结构总结.md
实验执行总结.md
```

仍必须扫描所有 `*.md`。

对于任何与 activation site 数量有关的：

```text
九个
9-site
9 sites
× 9
nine sites
```

全部更新。

但不要修改与 site topology 无关的数字 9。

---

# 26. 代码注释 / Docstring / Error Message 同样要清理

不仅 Markdown，Python 内的旧文字也要搜索。

修改所有类似：

```text
nine sites
9 replacement sites
Expected nine sites per layer
× 9 replacement sites
```

改为基于：

```python
SITE_COUNT
```

或统一的 10-site 表述。

目标是：

```bash
rg -n '九个|nine sites|9 sites|9 replacement sites|Expected nine' .
```

不再返回任何仍然描述旧 topology 的内容。

---

# 27. 不允许做的事情

本任务中禁止：

1. 新增 R5；
2. 修改 R1/R2/R3/R4 数学定义；
3. 把新 Site 9 放到 R4 后；
4. 把新 Site 9 放到 elementwise product 后；
5. 只记录 Site 9 activation 而不记录 GIF saliency；
6. 让 Site 9 activation 与 Site 10 saliency 错配；
7. 继续让原 `post_mlp_product_r4` 使用 site index 9；
8. 在 evaluation 中继续乘 9；
9. 只修改 `experiment_matrix.yaml` 而不重新 materialize generated configs；
10. 只修改 Python 而不更新 Markdown；
11. 静默接受旧 9-site calibration；
12. 静默删除旧 artifact；
13. 修改 Prefix 为 input_ids prepend；
14. 因本任务改动 unrelated training/checkpoint 逻辑；
15. 用全局字符串替换方式误改所有数字 9。

---

# 28. 推荐实施顺序

按以下顺序修改，降低中间状态出错概率。

## Step 1：新增统一 topology

创建：

```text
snn2/sites.py
```

定义：

```text
SITE_TOPOLOGY_VERSION
SITE_NAMES
SITE_COORDINATES
SITE_IDS
SITE_COUNT
site_key
expected_site_dirnames
topology_metadata
```

---

## Step 2：迁移 `stats.py`

去掉 site topology 的重复定义。

改为从 `sites.py` 导入。

---

## Step 3：修改 MLP integration

在：

```text
up_proj 后
```

增加：

```python
controller.apply(layer_index, 9, up)
```

原 product Site 9 改为 Site 10。

同步 saliency：

```text
Site 8 ← gate² up²
Site 9 ← gate² up²
Site 10 ← down_proj consumer
```

---

## Step 4：修改 config

```text
expected_sites_per_layer: 10
```

并在 `config.py` 验证与 `SITE_COUNT` 一致。

---

## Step 5：修改 calibration / manifest / stale protection

- exact 10-site validation；
- topology metadata；
- legacy 9-site fail-fast。

---

## Step 6：修改 conversion / verify

全部基于统一 topology。

---

## Step 7：修改 evaluation

所有：

```text
× 9
```

改为：

```python
* SITE_COUNT
```

---

## Step 8：修改测试

加入新 topology、MLP placement、identity parity、saliency、legacy rejection 测试。

---

## Step 9：重新生成 config

```bash
python scripts/materialize_configs.py
```

---

## Step 10：更新所有 Markdown

尤其：

```text
代码结构总结.md
实验执行总结.md
```

---

## Step 11：全仓库清理检查

执行：

```bash
rg -n \
  '九个|9 sites|9 site|nine sites|Expected nine|9 replacement sites|× 9|site_09_post_mlp_product_r4' \
  .
```

逐条检查剩余命中是否合法。

注意：

`site_09_post_mlp_product_r4` 可以出现在**专门用于检测 legacy artifact 的代码 / 测试**中。

除此之外，不应再作为当前 topology 的合法 site 名出现。

---

# 29. 最终验证命令

至少执行：

```bash
python scripts/materialize_configs.py
pytest -q
```

然后：

```bash
rg -n 'expected_sites_per_layer' \
  configs/experiment_matrix.yaml \
  configs/generated
```

必须全部为：

```text
10
```

再执行：

```bash
rg -n \
  '九个|nine sites|9 sites|9 replacement sites|Expected nine' \
  .
```

不应存在仍把当前项目描述为 9-site 的内容。

检查当前 topology：

```bash
python - <<'PY'
from snn2.sites import (
    SITE_COUNT,
    SITE_IDS,
    SITE_NAMES,
    SITE_COORDINATES,
    SITE_TOPOLOGY_VERSION,
)

print("version:", SITE_TOPOLOGY_VERSION)
print("count:", SITE_COUNT)

for i in SITE_IDS:
    print(i, SITE_NAMES[i], SITE_COORDINATES[i])

assert SITE_TOPOLOGY_VERSION == 2
assert SITE_COUNT == 10
assert SITE_IDS == tuple(range(1, 11))
assert SITE_NAMES[9] == "post_mlp_up_proj"
assert SITE_COORDINATES[9] == "I"
assert SITE_NAMES[10] == "post_mlp_product_r4"
assert SITE_COORDINATES[10] == "R4"
PY
```

---

# 30. 最终验收标准

只有同时满足以下条件，任务才算完成。

- [ ] 项目有且只有一套 authoritative 10-site topology；
- [ ] `SITE_COUNT == 10`；
- [ ] Site 1–8 语义保持不变；
- [ ] 新 Site 9 位于 `mlp.up_proj(x)` 之后；
- [ ] Site 9 在 gate-up product 之前；
- [ ] Site 9 不经过额外 rotation；
- [ ] Site 9 coordinate = `I`；
- [ ] 原 Site 9 `post_mlp_product_r4` 已完整顺延为 Site 10；
- [ ] Site 10 coordinate = `R4`；
- [ ] Site 8 / Site 9 都有 product-aware GIF saliency；
- [ ] Site 10 使用 down_proj linear-consumer saliency；
- [ ] Phase/GIF training 会在 Site 8、9、10 正确 replacement；
- [ ] Phase/GIF/MTN deployment 会在 Site 9 正确参与 temporal path；
- [ ] calibration 每层生成 exactly 10 个 site；
- [ ] legacy 9-site calibration 会被明确拒绝；
- [ ] statistics/calibration/conversion/verification metadata 记录 topology version / site count；
- [ ] evaluation operator count 使用 `SITE_COUNT`；
- [ ] `configs/experiment_matrix.yaml` 为 10；
- [ ] 全部 12 个 `configs/generated/*.yaml` 为 10；
- [ ] `代码结构总结.md` 已统一为 10-site；
- [ ] `实验执行总结.md` 已统一为 10-site；
- [ ] 其它 Markdown 中不存在旧 9-site 当前态描述；
- [ ] Python comments/docstrings/error messages 中不存在旧 9-site 当前态描述；
- [ ] Prefix 行为未改变；
- [ ] Rotation 数学与 fusion 行为未改变；
- [ ] Softmax Site 5 variable-length calibration 行为未改变；
- [ ] `pytest -q` 通过；
- [ ] 最终 `git diff` 只包含本任务需要的改动和由 `materialize_configs.py` 生成的配置同步。

---

# 31. 修改完成后 Codex 应给出的结果摘要

完成代码后，在最终回复中至少报告：

1. 修改了哪些文件；
2. 新增了哪些文件；
3. 新 Site 9 的准确位置；
4. 原 Site 9 如何迁移为 Site 10；
5. Site 8/9/10 的 saliency 如何处理；
6. 哪些硬编码 9 被替换为统一 `SITE_COUNT`；
7. calibration 如何拒绝旧 9-site artifact；
8. 哪些 Markdown 被更新；
9. `python scripts/materialize_configs.py` 是否成功；
10. `pytest -q` 的结果；
11. 是否还存在旧 9-site 描述的 `rg` 命中，以及这些命中为什么合法（例如 legacy compatibility test）。

如果测试失败，不要声称任务完成；应明确列出失败项和原因。

