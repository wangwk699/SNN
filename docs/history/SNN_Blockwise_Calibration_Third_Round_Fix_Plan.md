# SNN Block-wise SNN-aware Calibration 第三轮修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 当前审查基准 commit：`8ac7a991e679c27fcf5650c6ebe3376ae054f491`  
> 本文档针对第二轮修复完成后的剩余问题进行收尾。  
> 服务器上的 Codex 应当在**没有其它对话上下文**的情况下，仅凭本文档完成代码修改、测试和验证。

---

# 1. 本轮总体结论

第二轮修改之后，之前阻断 `*_previous_layers_snn=true` 的主要 P0 问题已经基本修复：

- temporal bootstrap 已使用 `T*B`；
- direct decoder layer 已使用 `past_key_value`；
- `SiteController` 已绑定 `site_root`；
- target state 已使用真实 `site_key()` 路径；
- Site 3/4 sequential collect 已恢复 native attention-head layout；
- GIF saliency 已支持 `calibration_collect`；
- Prefix positions 已重新按 legacy ANN calibration 规则排除；
- tuple/list/dict Tensor 已支持递归 device/CPU transfer；
- current-layer module cache 已支持清理；
- Stage A trajectory metadata 已基本统一；
- vanilla-analysis path 已固定为 effective all-false trajectory。

因此本轮**不要重构当前 block-wise 主执行链**。

本轮只处理剩余三个问题：

1. **stale target-specific statistics/state 可能被旧 artifact 静默污染；**
2. **runtime provenance validator 仍通过 `T=1` 绕过真实 Phase/MTN trajectory 校验；**
3. **测试仍未真正覆盖完整 `collect_blockwise_snn_conditioned_statistics()` E2E 主链路。**

---

# 2. 本轮绝对不能改变的行为

以下现有行为全部保留：

```text
ANN common statistics
    ↓
target-specific blockwise trajectory
    ↓
Block l collect/bypass
    ↓
<neuron>_statistics.pt
    ↓
materialize current Block l target state
    ↓
clear Block l module cache
    ↓
Block l real Temporal SNN deploy
    ↓
Block l+1 input
```

保留：

- 三个独立开关；
- Phase/GIF/MTN trajectory 彼此独立；
- strict block-wise，而不是 site-wise；
- GIF=true 时整套 GIF qparams + saliency + masks 重算；
- common Clip 数学规则；
- Prefix discovery；
- rotation；
- Site topology；
- Site 2 all-low；
- Site 5 GIF identity；
- Site 6 merged head；
- Site 8/9 GIF identity；
- Phase/MTN/GIF neuron 数学；
- ANN fine-tuning 路径。

---

# 3. P1：清理 stale target-specific artifacts，禁止旧结果污染 rerun

## 3.1 当前风险

当前 block-wise runner 的 target statistics 保存逻辑大致为：

```python
for key, stats in store.items.items():
    torch.save(
        stats.state_dict(),
        root / key / f"{neuron}_statistics.pt",
    )
```

而：

```python
materialize_target_state(...)
```

只检查：

```python
if not source.exists():
    raise FileNotFoundError(source)
```

问题在于：

如果同一个 trajectory artifact 目录之前跑过一次，但中途失败，例如留下：

```text
layer_010/site_06_post_attention_merge/gif_statistics.pt
```

本次重新跑时，如果某个 hook 因 regression 没有重新记录该 site：

```text
本轮 store 中没有 Site 6
```

旧的：

```text
gif_statistics.pt
```

仍然存在。

随后：

```python
materialize_target_state()
```

看到文件存在，就会把旧文件误当作本轮结果。

这违反本项目要求的：

```text
fail-fast
```

也会造成最危险的一类 bug：

> calibration 成功结束，但某些 layer/site 实际来自旧实验 artifact。

---

# 4. 推荐修复策略：每个 target trajectory 启动时先清空该 target 的派生 artifact

推荐实现一个公开 helper，例如放在：

```text
snn2/blockwise_calibration.py
```

或者：

```text
snn2/calibration.py
```

例如：

```python
def clear_target_trajectory_artifacts(
    site_root: str | Path,
    neuron: str,
) -> None:
    ...
```

只删除当前 neuron 对应的 target-specific 文件。

---

## 4.1 删除范围

对于：

```text
neuron="gif"
```

删除：

```text
layer_*/site_*/gif_statistics.pt
layer_*/site_*/gif_state.pt
```

Phase：

```text
phase_statistics.pt
phase_state.pt
```

MTN：

```text
mtn_statistics.pt
mtn_state.pt
```

以及 Final RMSNorm：

```text
_global/final_rmsnorm/phase_statistics.pt
_global/final_rmsnorm/phase_state.pt
```

或：

```text
_global/final_rmsnorm/mtn_statistics.pt
_global/final_rmsnorm/mtn_state.pt
```

GIF 不需要 global state。

---

## 4.2 不允许删除 common ANN statistics

严禁删除：

```text
statistics.pt
```

因为：

```text
phase_previous_layers_snn=false
gif_previous_layers_snn=false
mtn_previous_layers_snn=false
```

对应 state 仍然需要 common ANN statistics。

---

## 4.3 不允许删除其它 neuron 的 target files

如果当前跑 GIF：

```text
不要删除 phase_statistics.pt
不要删除 mtn_statistics.pt
```

因为三条 trajectory 独立，并且脚本可能在同一次 Stage A 中按顺序运行多个 target。

---

# 5. 清理时机

在：

```python
collect_blockwise_snn_conditioned_statistics(...)
```

开始时：

```python
clear_target_trajectory_artifacts(
    site_root,
    neuron,
)
```

然后：

```python
controller.clear_runtime_module_cache()
controller.begin_sequential_calibration(...)
```

---

## 5.1 为什么不只在每个 block 前删除

可以每 block 前删，但更简单、可靠的做法是：

```text
trajectory 开始时清空当前 target 的全部旧文件
```

这样：

- interrupted rerun 不会误用旧结果；
- 最终缺失 site 会自然 fail；
- 逻辑更清晰。

---

# 6. 每个 block collect 后必须验证 topology 完整

仅清 stale 文件仍不够。

每个 block collect 后，在 `_save_target_statistics()` 前必须验证：

```text
当前 StatisticsStore 中恰好包含当前 block 的 10 个 site
```

假设：

```python
expected = {
    site_key(layer_index, site_index)
    for site_index in SITE_IDS
}
```

实际：

```python
actual = set(store.items)
```

必须：

```python
actual == expected
```

否则：

```python
missing = expected - actual
unexpected = actual - expected
raise RuntimeError(...)
```

推荐报错：

```text
Blockwise calibration site coverage mismatch:
neuron=gif
layer=7
missing=[...]
unexpected=[...]
```

---

# 7. `_save_target_statistics()` 改为显式传 layer_index

当前建议修改为：

```python
def _save_target_statistics(
    store: StatisticsStore,
    root: Path,
    neuron: Neuron,
    *,
    layer_index: int | None = None,
    global_only: bool = False,
) -> None:
    ...
```

普通 block：

```python
_save_target_statistics(
    controller.statistics,
    root,
    neuron,
    layer_index=index,
)
```

Final RMSNorm：

```python
_save_target_statistics(
    controller.statistics,
    root,
    neuron,
    global_only=True,
)
```

---

## 7.1 普通 block validation

必须检查：

```python
expected = {
    site_key(layer_index, site_index)
    for site_index in SITE_IDS
}
actual = set(store.items)
```

必须完全一致。

同时：

```python
store.global_items
```

在普通 block collect 阶段应该为空。

如果不为空：

```python
raise RuntimeError(...)
```

防止 Final RMSNorm hook 在 direct block run 中被意外触发。

---

## 7.2 Final RMSNorm validation

当：

```text
global_only=True
```

必须：

```python
set(store.global_items) == {"final_rmsnorm"}
```

同时：

```python
store.items
```

必须为空。

---

# 8. 保存后立即重新检查文件是否全部写出

普通 layer：

```python
for site_index in SITE_IDS:
    path = root / site_key(layer_index, site_index) / f"{neuron}_statistics.pt"
    if not path.exists():
        raise FileNotFoundError(path)
```

然后再：

```python
materialize_target_state(...)
```

---

# 9. materialize_target_state() 继续保持 fail-fast

当前：

```python
if not source.exists():
    raise FileNotFoundError(source)
```

这一点保持。

不要改回：

```python
continue
```

---

# 10. State 文件也要避免旧 cache

当前已经：

```python
controller.clear_layer_module_cache(index)
```

这一点保留。

另外 trajectory 启动时清除：

```text
<neuron>_state.pt
```

确保任何 state 都一定是本轮 statistics 新生成。

---

# 11. P1/P2：修复 runtime provenance validator 的 `T=1` 绕过

## 11.1 当前问题

当前：

```text
snn2/state_validation.py
```

仍通过：

```python
PhaseSurrogate(state, T=1)
```

和：

```python
MultiThresholdNeuron(
    state,
    T=1,
    K=1,
    threshold_factor=0.75,
)
```

来做 state validation。

同时 neuron constructor 里又有特殊逻辑：

```python
if state.get("previous_layers_snn") is True:
    if self.T != 1 and recorded != self.T:
        raise ...
```

MTN 类似：

```python
if self.T != 1 and expected != actual:
    raise ...
```

于是：

```text
T=1
```

变成 validator 的特殊逃生口。

这会导致：

```text
state 声称 calibration_phase_T=4
validator 用 T=1
仍然通过
```

validator 没有真正验证 runtime provenance。

---

# 12. 正确设计：schema validation 与 runtime validation 分离

不要继续使用：

```text
构造一个 fake neuron(T=1)
```

来完成所有 state 校验。

推荐新增纯 schema validator。

例如放在：

```text
snn2/neurons.py
```

或：

```text
snn2/state_validation.py
```

---

## 12.1 Phase

新增：

```python
def validate_phase_state_schema(
    state: dict[str, Any],
) -> None:
    ...
```

只检查：

- `kind == "phase"`
- format version
- tau calibration metadata
- layout
- parameter shape
- clamp metadata
- tau finite
- tau in valid range
- trajectory metadata 基础结构

不检查 runtime T。

---

## 12.2 MTN

新增：

```python
def validate_mtn_state_schema(
    state: dict[str, Any],
) -> None:
    ...
```

只检查：

- state header
- base_scale calibration metadata
- layout
- tensor shape
- finite
- clamp
- trajectory metadata 基础结构

---

## 12.3 GIF

GIF 目前 constructor 本身没有 config runtime ambiguity，因为：

```text
GIF_LOCAL_STEPS
```

是固定 implementation constant。

可以继续：

```python
gif_module_from_state(state)
```

但如果已有公共 pure validator，也可以一起统一。

---

# 13. Neuron constructor 应严格校验真实 runtime

修改：

```python
PhaseSurrogate.__init__()
```

当前：

```python
if state.get("previous_layers_snn") is True:
    recorded = state.get("calibration_phase_T")
    if not isinstance(recorded, int) or (
        self.T != 1 and recorded != self.T
    ):
        raise ValueError(...)
```

改成：

```python
if state.get("previous_layers_snn") is True:
    recorded = state.get("calibration_phase_T")
    if not isinstance(recorded, int) or recorded != self.T:
        raise ValueError(
            "Phase sequential-calibration T provenance mismatch"
        )
```

彻底删除：

```text
self.T != 1
```

例外。

---

# 14. MTN constructor 同样严格 equality

当前：

```python
if self.T != 1 and expected != actual:
```

改成：

```python
if expected != actual:
    raise ValueError(
        "MTN sequential-calibration runtime provenance mismatch"
    )
```

其中：

```python
expected = (
    state["calibration_mtn_T"],
    state["calibration_mtn_K"],
    state["calibration_mtn_threshold_factor"],
)

actual = (
    self.T,
    self.K,
    self.threshold_factor,
)
```

---

# 15. `validate_site_state_bundle()` 必须支持 cfg

推荐 signature：

```python
def validate_site_state_bundle(
    site_root: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    clip_policy: ClipBundlePolicy,
    expected_num_hidden_layers: int | None = None,
) -> dict[str, Any]:
```

---

# 16. `cfg is None` 时

做纯 schema validation：

```python
validate_phase_state_schema(state)
validate_mtn_state_schema(state)
gif_module_from_state(...)
```

不能 fake：

```text
Phase T=1
MTN T=1/K=1
```

---

# 17. `cfg is not None` 时额外做 runtime validation

对于 Phase state：

```python
if state["previous_layers_snn"]:
    assert state["calibration_phase_T"] == int(cfg["phase"]["T"])
```

MTN：

```python
if state["previous_layers_snn"]:
    assert state["calibration_mtn_T"] == int(cfg["mtn"]["T"])
    assert state["calibration_mtn_K"] == int(cfg["mtn"]["K"])
    assert state["calibration_mtn_threshold_factor"] == float(cfg["mtn"]["threshold_factor"])
```

GIF：

```python
if state["previous_layers_snn"]:
    assert state["calibration_gif_temporal_steps"] == GIF_LOCAL_STEPS
```

---

# 18. Manifest 与 state 必须互相一致

对于每个 state：

```text
manifest effective_previous_layers_snn[neuron]
```

必须等于：

```text
state["previous_layers_snn"]
```

如果 manifest：

```json
"gif": true
```

但某个：

```text
gif_state.pt
```

写着：

```json
"previous_layers_snn": false
```

直接 fail。

---

# 19. training / conversion / deployment validation 全部传 cfg

以下位置调用：

```python
validate_site_state_bundle(...)
```

时，如果当前函数已有 `cfg`，应传入：

```python
cfg=cfg
```

重点检查：

```text
snn2/training.py
snn2/conversion.py
snn2/controller.py
evaluation / conversion validation
```

---

# 20. `SiteController.set_deployment()` 也应能够真实验证 cfg

当前 controller 自身只持有：

```text
phase_T
mtn_T
mtn_K
mtn_threshold_factor
```

如果不方便传完整 cfg，可有两种方案。

推荐方案 A：

让 controller 初始化时接受：

```python
calibration_runtime_signature={
    "phase_T": ...,
    "mtn_T": ...,
    "mtn_K": ...,
    "mtn_threshold_factor": ...,
}
```

但这会增加改动。

更简单的方案 B：

保留 `set_deployment()` bundle schema validation；

真实 runtime mismatch 由：

```python
_load()
```

构造 neuron 时严格 equality 自动 fail。

这样也可以。

本轮不要求为了 validator 再大改 controller API。

---

# 21. Final RMSNorm states 同样要真实验证

当前：

```text
_global/final_rmsnorm/phase_state.pt
_global/final_rmsnorm/mtn_state.pt
```

也必须使用纯 schema validator。

如果传 cfg：

```text
Phase calibration T
MTN calibration T/K/factor
```

也要检查。

---

# 22. P1/P2：补真正的 blockwise E2E test

当前：

```text
tests/test_blockwise_calibration_runtime.py
```

只覆盖：

- temporal Site 3/4 layout；
- Prefix exclusion；
- canonical target state path；
- temporal input preparation。

这还不够。

必须新增至少一个测试，真实调用：

```python
collect_blockwise_snn_conditioned_statistics(...)
```

---

# 23. E2E test 不需要加载真实 8B 模型

不要让单元测试依赖：

```text
Qwen3-8B
Llama3-8B
GPU
Hugging Face network
```

构造 toy model。

但 toy model 必须尽可能遵守当前 runner 依赖接口：

```text
model.model.embed_tokens
model.model.layers
model.model.norm
model.lm_head
model.config.num_hidden_layers
```

或者使用当前 `get_model_parts()` 支持的结构。

---

# 24. 推荐 toy model

至少：

```text
2 decoder blocks
hidden_dim=4
sequence_length=3
batch_size=1
```

每个 toy block 可以非常简单：

```python
class ToyBlock(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        ...
        return (hidden_states_out,)
```

但为了真正测试 SiteController integration，最好复用一个 minimal attention/MLP 模块，并安装：

```python
install_model_integration(...)
```

如果实现过于复杂，也至少确保：

```text
collect_blockwise_snn_conditioned_statistics
```

真实运行，且 `materialize_target_state` / module cache / sequential deploy 都发生。

---

# 25. E2E 必测场景 1：完整 GIF trajectory

配置：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: true
mtn_previous_layers_snn: false
```

真实调用：

```python
collect_blockwise_snn_conditioned_statistics(
    ...,
    neuron="gif",
)
```

最终必须验证：

```text
Block 0 target stats written
Block 0 gif_state written
Block 0 temporal deploy executed
Block 1 target stats written
Block 1 target stats differs from ANN baseline in controlled toy case
```

---

# 26. E2E 必测场景 2：fresh state 确实用于下一层 trajectory

这是最重要的测试。

构造 toy neuron/state，使：

```text
Block 0 ANN output
```

与：

```text
Block 0 GIF temporal output
```

有确定、明显差异。

然后验证：

```text
Block 1 gif_statistics
```

对应：

```text
Block 0 GIF output
```

而不是：

```text
Block 0 ANN output
```

---

# 27. E2E 必测场景 3：stale artifact 被清理

测试流程：

1. 预先手动创建：

```text
layer_001/site_xxx/gif_statistics.pt
```

内容设为明显假的：

```text
999
```

2. 运行 blockwise GIF calibration；

3. 故意让本次某个 site 不被观察。

预期：

```text
不是复用旧文件
而是 fail-fast
```

---

# 28. 更简单的 stale artifact 单测

如果完整 E2E 中不好构造 missing site，可直接单测：

```python
clear_target_trajectory_artifacts(...)
```

确保：

```text
gif_statistics.pt
gif_state.pt
```

被删除，而：

```text
statistics.pt
phase_statistics.pt
mtn_statistics.pt
```

保留。

同时 E2E 再验证 site coverage mismatch。

---

# 29. E2E 必测场景 4：module cache freshness

流程：

1. 写入 `gif_state.pt` A；
2. controller `_load()`；
3. 覆盖成 `gif_state.pt` B；
4. `clear_layer_module_cache(layer)`；
5. 再 deploy；
6. 输出必须对应 B。

证明：

```text
current block 新生成 state
```

不会被旧 Python module object 污染。

---

# 30. E2E 必测场景 5：Phase runtime mismatch fail

构造 state：

```json
{
  "previous_layers_snn": true,
  "calibration_phase_T": 4
}
```

然后：

```python
PhaseSurrogate(state, T=3)
```

必须 fail。

之前的：

```text
T=1 validator escape
```

必须彻底消失。

---

# 31. E2E 必测场景 6：MTN runtime mismatch fail

state：

```text
calibration_mtn_T=4
calibration_mtn_K=6
calibration_mtn_threshold_factor=0.75
```

任意一个 runtime 不同：

```text
T=3
K=5
factor=0.5
```

都必须 fail。

---

# 32. E2E 必测场景 7：schema validation 不需要 fake T

调用：

```python
validate_site_state_bundle(...)
```

在：

```text
cfg=None
```

情况下，应成功验证合法 sequential Phase/MTN state。

不能内部构造：

```text
T=1
```

来“绕过”。

---

# 33. E2E 必测场景 8：manifest ↔ state flag mismatch

manifest：

```json
"effective_previous_layers_snn": {
  "phase": false,
  "gif": true,
  "mtn": false
}
```

如果任意：

```text
gif_state.pt
```

中：

```json
"previous_layers_snn": false
```

必须 fail。

---

# 34. E2E 必测场景 9：Final RMSNorm provenance

Phase=true：

```text
_global/final_rmsnorm/phase_state.pt
```

必须：

```text
previous_layers_snn=true
calibration_phase_T == cfg.phase.T
```

MTN 同理。

---

# 35. E2E 必测场景 10：all-false regression

当：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: false
mtn_previous_layers_snn: false
```

必须：

```text
不调用 blockwise trajectory
所有 state source = statistics.pt
```

旧逻辑保持。

---

# 36. 建议修改文件

至少检查：

```text
snn2/blockwise_calibration.py
snn2/calibration.py
snn2/state_validation.py
snn2/neurons.py
snn2/training.py
snn2/conversion.py
snn2/controller.py
```

测试：

```text
tests/test_blockwise_calibration_runtime.py
tests/test_state_validation.py
tests/test_neurons.py
```

如当前仓库没有这些 test 文件，可新增。

---

# 37. 推荐实现顺序

## Step 1

实现：

```python
clear_target_trajectory_artifacts(...)
```

---

## Step 2

给 `_save_target_statistics()` 增加：

```text
layer_index
global_only
```

以及严格 site coverage validation。

---

## Step 3

trajectory 开始时清 stale target stats/state。

---

## Step 4

拆纯：

```text
validate_phase_state_schema
validate_mtn_state_schema
```

---

## Step 5

删除：

```text
self.T != 1
```

validator bypass。

---

## Step 6

让：

```python
validate_site_state_bundle(..., cfg=cfg)
```

支持真实 runtime provenance。

---

## Step 7

更新 training / conversion validation 调用。

---

## Step 8

补完整 E2E runtime tests。

---

# 38. 真实 smoke test

代码修改和 `pytest -q` 全通过后，再做真实小样本 smoke test。

建议：

```yaml
calibration:
  num_samples: 2
  phase_previous_layers_snn: false
  gif_previous_layers_snn: true
  mtn_previous_layers_snn: false
```

运行：

```bash
python scripts/calibrate_sites.py \
  --config "$CFG" \
  --stage ann_training \
  --calibration-phase A
```

---

# 39. Smoke test 前故意制造 stale artifact

为了验证本轮 P1 修复，可先在目标 trajectory 路径下人为建立一个假的：

```text
layer_000/site_01_*/gif_statistics.pt
```

或者使用上一轮失败残留。

然后重新运行 Stage A。

预期：

```text
旧 gif_statistics.pt 在 trajectory 开始时被删除
```

而不是复用。

---

# 40. Smoke test artifact 检查

最终检查随机：

```text
layer_000
layer_001
最后一层
```

每个 site 应有：

```text
statistics.pt
gif_statistics.pt

phase_state.pt
gif_state.pt
mtn_state.pt
```

对于当前 F/T/F 配置：

```text
phase_state source -> statistics.pt
gif_state source   -> gif_statistics.pt
mtn_state source   -> statistics.pt
```

---

# 41. Manifest 检查

必须：

```json
"effective_previous_layers_snn": {
  "phase": false,
  "gif": true,
  "mtn": false
}
```

且：

```json
"calibration_trajectory": {
  "phase": {
    "previous_layers_snn": false,
    "source": "ann_common"
  },
  "gif": {
    "previous_layers_snn": true,
    "source": "sequential_temporal_gif",
    "gif_temporal_steps": 2
  },
  "mtn": {
    "previous_layers_snn": false,
    "source": "ann_common"
  }
}
```

---

# 42. 每个 GIF state 检查

```python
state["previous_layers_snn"] is True
state["calibration_trajectory"] == "sequential_temporal_gif"
state["calibration_gif_temporal_steps"] == 2
```

---

# 43. Phase/MTN false state 检查

例如：

```python
phase_state["previous_layers_snn"] is False
phase_state["calibration_trajectory"] == "ann_common"
```

不应携带：

```text
calibration_phase_T
```

MTN false 同理。

---

# 44. Final RMSNorm

当前 F/T/F：

```text
Phase false
MTN false
```

所以 Final RMSNorm Phase/MTN state 应来自：

```text
statistics.pt
```

而不是 target-specific stats。

GIF 无 final global state。

---

# 45. Failure policy

以下全部必须 fail-fast：

- 当前 block 缺一个 site；
- 当前 block 出现额外其它 layer site；
- stale target-specific stats 未清理；
- target state source 不存在；
- manifest/state previous_layers_snn 不一致；
- Phase calibration T 与 runtime 不一致；
- MTN calibration T/K/factor 与 runtime 不一致；
- GIF calibration temporal steps 与 implementation 不一致；
- Final RMSNorm provenance 不一致。

---

# 46. 本轮不要求的修改

不要再改：

- temporal attention 数学；
- Prefix statistics exclusion；
- Site 3/4 layout；
- GIF saliency formula；
- GIF mask selection；
- Phase math；
- MTN math；
- Clip math；
- artifact trajectory path；
- LR / scheduler；
- dataset；
- lm-eval；
- ANN fine-tuning memory optimization。

---

# 47. 最终验收标准

完成本轮后必须满足：

- [ ] rerun 不可能复用旧 target-specific statistics；
- [ ] rerun 不可能复用旧 target-specific state；
- [ ] 每个 block 必须收集完整 10-site topology；
- [ ] 缺任何 site 立即 fail；
- [ ] Final RMSNorm global statistics coverage 也严格检查；
- [ ] Phase sequential state runtime T mismatch 会 fail；
- [ ] MTN sequential state runtime T/K/factor mismatch 会 fail；
- [ ] validator 不再依赖 fake `T=1`；
- [ ] manifest effective flag 与每个 state 一致；
- [ ] training validation 使用真实 cfg；
- [ ] conversion validation 使用真实 cfg；
- [ ] 完整 E2E test 实际调用 `collect_blockwise_snn_conditioned_statistics()`；
- [ ] E2E test 证明 Block 1 target stats 来自 Block 0 Temporal SNN output；
- [ ] E2E test 证明 stale artifact 被清理；
- [ ] E2E test 证明 module cache freshness；
- [ ] all-false regression 不变；
- [ ] `pytest -q` 全部通过；
- [ ] GIF `num_samples=2` 真实 Stage A smoke test 成功。

---

# 48. 本轮完成后的预期状态

完成本轮后，这套 calibration pipeline 应具备：

```text
correct numerical trajectory
+
strict artifact isolation
+
strict stale-artifact prevention
+
strict runtime provenance
+
real end-to-end regression coverage
```

这时才适合作为正式 Tulu-3 / Llama-3：

```text
phase_aware
gif_aware
```

实验的长期稳定基线。
