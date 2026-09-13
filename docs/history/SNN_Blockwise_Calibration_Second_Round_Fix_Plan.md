# SNN Block-wise SNN-aware Calibration 第二轮修复方案

> 目标仓库：`https://github.com/wangwk699/SNN/tree/main`  
> 本文档针对当前 `main` 分支中已经完成的第一轮 block-wise SNN-aware calibration 实现进行修复。  
> 审查基准 commit：`5cac680b8759f4e167403f1110e482deab9d0180`。  
> 本文档必须作为**独立实施说明**使用：服务器上的 Codex 不需要任何额外对话上下文，仅依据本文档即可完成修改、测试和验证。

---

# 1. 本轮修改目标

第一轮已经完成了以下外围框架：

- `calibration.phase_previous_layers_snn`
- `calibration.gif_previous_layers_snn`
- `calibration.mtn_previous_layers_snn`
- trajectory-dependent artifact path
- Stage A / Stage B provenance
- training provenance
- conversion provenance
- target-specific `*_statistics.pt`
- block-wise calibration runner 框架
- schema/version bump

这些方向**保留**。

但是当前实现中，只要任意：

```yaml
phase_previous_layers_snn: true
gif_previous_layers_snn: true
mtn_previous_layers_snn: true
```

block-wise sequential calibration 主执行链路仍存在多个阻断性问题，导致：

- temporal bootstrap 直接报错；
- Prefix direct-layer forward 不正确；
- 当前 block 新 state 没有正确 materialize；
- Site 3/4 statistics shape 错误；
- GIF saliency 缺失；
- current block propagation 不能真正使用刚刚校准的 neuron state；
- multi-GPU 下可能发生 position embedding device mismatch；
- manifest 中 runtime dependency 与真实 trajectory 不一致。

本轮只修复这些问题，不修改已经确认的 calibration 数学定义和实验策略。

---

# 2. 本轮必须保留的既有设计

以下内容已经确认，禁止改回。

## 2.1 三个独立开关

继续使用：

```yaml
calibration:
  phase_previous_layers_snn: false
  gif_previous_layers_snn: false
  mtn_previous_layers_snn: false
```

## 2.2 严格 block-wise

对于目标神经元 \(n\)：

\[
X_l^{(n)} =
F_{l-1}^{(n)} \circ \cdots \circ F_0^{(n)}(X_0).
\]

执行顺序必须是：

```text
Block 0 collect
→ materialize Block 0 target state
→ Block 0 Temporal SNN deploy
→ 得到 Block 1 input

Block 1 collect
→ materialize Block 1 target state
→ Block 1 Temporal SNN deploy
→ 得到 Block 2 input
```

同一 Transformer block 内所有 sites 同时 collect。

禁止 site-wise：

```text
Site 1 校准
→ 插入 Site 1 neuron
→ Site 2 校准
```

## 2.3 三条 trajectory 独立

例如：

```yaml
phase_previous_layers_snn: false
gif_previous_layers_snn: true
mtn_previous_layers_snn: false
```

则：

```text
Phase -> ANN common statistics
GIF   -> GIF temporal sequential statistics
MTN   -> ANN common statistics
```

不能串 trajectory。

## 2.4 GIF=true 时 GIF state 整体重算

必须全部基于 GIF-SNN-conditioned trajectory：

- value min/max
- low scale/zero
- high scale/zero
- saliency
- role-specific saliency
- mask

## 2.5 common Clip 数学规则不变

Stage B：

```python
build_clip_state(
    phase_state,
    gif_state,
    mtn_state,
    ...
)
```

保持当前实现。

---

# 3. 本轮 P0-1：修复 temporal bootstrap

## 3.1 当前错误

当前：

```text
snn2/blockwise_calibration.py
```

在 `_bootstrap()` 前执行：

```python
controller.begin_sequential_calibration(neuron, 0)
```

此时：

```python
controller.temporal_execution_enabled == True
```

但是 `_bootstrap()` 仍用普通 batch：

```python
model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    use_cache=False,
)
```

并没有扩展为：

```text
[T * B, L]
```

当前 temporal embedding hook 会检查：

```python
output.shape[0] % steps == 0
```

而 calibration batch size 固定为 1，所以 GIF `T=2`、Phase `T=4` 都会失败。

## 3.2 正确行为

block-wise bootstrap 必须真正按 temporal deployment 语义构造 first-layer input：

```text
input_ids      [B, L]
attention_mask [B, L]
```

扩展为：

```text
input_ids_temporal      [T*B, L]
attention_mask_temporal [T*B, L]
```

使用 time-major 排列。

## 3.3 不要重复实现 temporal input preparation

建议从：

```text
snn2/model_integration.py
```

抽出公共 helper，例如：

```python
def prepare_temporal_model_inputs(
    input_ids,
    attention_mask,
    *,
    steps,
    **kwargs,
):
    ...
```

由：

```python
temporal_forward()
```

和：

```python
blockwise calibration bootstrap
```

共同使用。

至少统一：

- repeated input IDs
- repeated attention mask
- position_ids expansion
- cache_position contract

## 3.4 Bootstrap 推荐实现

例如：

```python
@torch.no_grad()
def _bootstrap(model, loader, controller):
    steps = int(controller.temporal_steps)

    for batch in loader:
        repeated_ids = batch["input_ids"].repeat(steps, 1)
        repeated_mask = batch["attention_mask"].repeat(steps, 1)

        try:
            model(
                input_ids=repeated_ids.to(device),
                attention_mask=repeated_mask.to(device),
                use_cache=False,
            )
        except _CatchFirstLayer:
            pass
```

## 3.5 Bootstrap 输出必须已经包含 temporal embedding policy

当前 temporal policy：

```text
uniform_embedding_divide_by_T
```

因此 catcher 抓到的：

```text
hidden_states
```

必须已经是：

```text
[T*B, L, D]
```

且 embedding 已执行 `/T`。

---

# 4. 本轮 P0-2：修复 DecoderLayer Prefix 参数名

## 4.1 当前错误

`blockwise_calibration._run()` 当前：

```python
kwargs["past_key_values"] = cache
```

但项目固定：

```text
transformers==4.53.2
```

LlamaDecoderLayer / Qwen3DecoderLayer 参数名都是：

```python
past_key_value
```

## 4.2 修改

必须改成：

```python
kwargs["past_key_value"] = cache
```

并在 bootstrap catcher 中继续排除：

```text
past_key_value
past_key_values
```

避免缓存 mutable cache。

## 4.3 每次 direct block forward 都创建 fresh cache

保持：

```python
fresh_prefix_dynamic_cache(...)
```

但每次：

```text
sample × block × collect/deploy
```

都必须重新创建。

---

# 5. 本轮 P0-3：SiteController 必须绑定 site_root

## 5.1 当前错误

`scripts/calibrate_sites.py` 创建 controller 时没有：

```python
site_root=site_root
```

sequential deploy 时 `_load()` 无法读取 target state。

## 5.2 修改

改为：

```python
controller = SiteController(
    mode="collect",
    site_root=site_root,
    phase_T=int(cfg["phase"]["T"]),
    mtn_T=int(cfg["mtn"]["T"]),
    mtn_K=int(cfg["mtn"]["K"]),
    mtn_threshold_factor=float(cfg["mtn"]["threshold_factor"]),
)
```

Stage A sequential deployment 不使用 Stage B Clip，因此：

```text
clip_root = None
common_clip_enabled = False
```

保持不变。

---

# 6. 本轮 P0-4：修复 materialize_target_state 的 site 路径

## 6.1 当前错误

当前手工拼：

```python
site_01
site_02
...
```

但真实目录是：

```text
site_01_post_input_rmsnorm
site_02_q_post_rope_r3
site_03_k_post_rope_r3
...
```

## 6.2 必须使用统一 helper

使用：

```python
from .sites import SITE_IDS, site_key

directories = [
    root / site_key(layer_index, site_index)
    for site_index in SITE_IDS
]
```

## 6.3 缺失 target statistics 必须 fail-fast

当前：

```python
if not source.exists():
    continue
```

改为：

```python
if not source.exists():
    raise FileNotFoundError(source)
```

不能静默跳过。

---

# 7. 本轮 P0-5：修复 Site 3 / Site 4 temporal collect shape

## 7.1 当前 contract

Site 3 / 4 statistics 是：

```text
attention_head
```

logical shape：

```text
[B, H, L, D_head]
```

## 7.2 当前错误

temporal attention path 先 merge 为：

```text
[T*B, L, H*D]
```

再 `controller.apply()`，导致 statistics 变成 last-dim layout。

## 7.3 正确实现

在 `snn2/temporal_model.py` 中：

```python
if controller.collecting_statistics:
    controller.record_activation(layer_index, 3, native_key)
    controller.record_activation(layer_index, 4, native_value)
    # 不改变 key/value
else:
    # deployment 才 merge → neuron → restore
```

current block collect 时，Site 3/4 只 record，不运行 neuron。

Site 6 继续保持 merged-head：

```text
[B,L,H*D]
```

不要改回 per-head。

---

# 8. 本轮 P0-6：修复 sequential GIF saliency 收集

## 8.1 当前错误

当前 helper：

```python
def _record_saliency(...):
    if controller.mode != "collect":
        return
```

sequential 时 mode 是：

```text
calibration_collect
```

所以 saliency 被丢弃。

## 8.2 修改

改为：

```python
if not controller.collecting_statistics:
    return
```

更推荐让 `SiteController.record_saliency()` 自己负责判断。

## 8.3 必须覆盖

- Site 1: q/k/v
- Site 3: QK-based K saliency, FP64
- Site 4: PV-based V saliency, FP64
- Site 6: o_proj consumer, FP32
- Site 7: gate/up, FP32
- Site 10: down_proj consumer, FP32

---

# 9. Prefix statistics exclusion 必须与 legacy ANN 一致

本轮只改变 previous-layer trajectory，不改变 Prefix token 是否参与统计。

旧 ANN path 使用 `past_length` 排除 Prefix positions。

Temporal path 必须补回相同规则。

## 9.1 Site 3/4 activation

```python
statistics_key = key[..., past_length:, :] if past_length else key
statistics_value = value[..., past_length:, :] if past_length else value
```

## 9.2 Site 3 saliency

```python
if past_length:
    key_score = key_score[..., past_length:, :]
```

## 9.3 Site 4 saliency

```python
if past_length:
    value_score = value_score[..., past_length:, :]
```

## 9.4 Site 5

保持 legacy path 当前定义；如果旧 path 排除 Prefix key columns，temporal path 也必须一致。

---

# 10. Multi-GPU：递归迁移 kwargs

## 10.1 当前问题

`position_embeddings=(cos, sin)` 是 tuple Tensor。

当前 `_cpu()` / `_run()` 只处理顶层 Tensor，无法递归迁移。

## 10.2 新增 tree-map helper

```python
def _tree_map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map_tensors(x, fn) for x in value)
    if isinstance(value, list):
        return [_tree_map_tensors(x, fn) for x in value]
    if isinstance(value, dict):
        return {k: _tree_map_tensors(v, fn) for k, v in value.items()}
    return value
```

CPU：

```python
def _to_cpu(value):
    return _tree_map_tensors(value, lambda x: x.detach().cpu())
```

device：

```python
def _to_device(value, device):
    return _tree_map_tensors(value, lambda x: x.to(device))
```

---

# 11. Final RMSNorm sequential calibration

Phase / MTN target 在所有 blocks 完成 temporal propagation 后：

```text
final temporal hidden
→ temporal RMSNorm
→ calibration_collect
→ logical sum
→ target-specific final statistics
```

Phase：

```text
_global/final_rmsnorm/phase_statistics.pt
```

MTN：

```text
_global/final_rmsnorm/mtn_statistics.pt
```

GIF 仍不生成 global state。

---

# 12. 当前 block state materialization 与 module cache

每层必须：

```text
collect
→ save target statistics
→ materialize target state
→ clear current-layer module cache
→ sequential deploy
```

建议新增：

```python
def clear_layer_module_cache(self, layer_index: int) -> None:
    prefix = f"layer_{layer_index:03d}/"
    for key in list(self._modules):
        if key.startswith(prefix):
            del self._modules[key]
```

以及：

```python
def clear_runtime_module_cache(self) -> None:
    self._modules.clear()
```

不要在 blockwise engine 直接访问私有 `_modules`。

---

# 13. Stage A manifest runtime dependency 修复

当前 `calibration_provenance()` 处理得较正确，但 `materialize_calibration_states()` 又覆盖为固定 independence。

必须集中为唯一 helper。

建议：

```python
def stage_a_trajectory_metadata(cfg, *, effective=True):
    ...
```

由：

- `calibration_provenance()`
- `materialize_calibration_states()`
- training validation
- conversion validation

共同使用。

正确语义：

### phase=true

Stage A 依赖：

```text
phase.T
```

### mtn=true

Stage A 依赖：

```text
mtn.T
mtn.K
mtn.threshold_factor
```

### gif=true

记录：

```text
gif_temporal_steps = GIF_LOCAL_STEPS
```

不能继续无条件声明这些字段 independent。

---

# 14. Vanilla analysis path

设计上：

```text
vanilla_analysis = always ANN-only
```

所以其 path 应使用 effective false trajectory，而不是 requested flags。

建议：

```python
calibration_trajectory_dirname(cfg, effective=False)
```

避免相同 vanilla analysis artifact 因请求的 true/false 被重复保存。

---

# 15. `build_gif_state()` 应真正独立

当前：

```python
def build_gif_state(statistics, cfg):
    return build_site_states(statistics, cfg)["gif"]
```

会无意义构造 Phase/MTN state。

建议真正抽出 GIF builder，使：

```python
materialize_target_state(..., neuron="gif")
```

只构建 GIF。

---

# 16. 推荐主流程

```python
def collect_blockwise_snn_conditioned_statistics(...):
    validate target
    validate batch_size == 1

    controller.clear_runtime_module_cache()
    controller.begin_sequential_calibration(neuron, 0)

    cached = bootstrap_temporal_inputs(...)

    for layer_idx, layer in enumerate(layers):
        # A. collect
        controller.begin_sequential_calibration(neuron, layer_idx)
        controller.statistics = StatisticsStore()

        for item in cached:
            run_decoder_layer(layer, item)

        save_target_statistics(...)

        # B. materialize current block state
        materialize_target_state(
            root,
            cfg,
            neuron,
            layer_index=layer_idx,
        )

        controller.clear_layer_module_cache(layer_idx)

        # C. deploy current block SNN
        controller.begin_sequential_deployment(layer_idx)

        next_cached = []
        for item in cached:
            output = run_decoder_layer(layer, item)
            next_cached.append(
                _LayerInput(
                    hidden_states=output.detach().cpu(),
                    kwargs=item.kwargs,
                )
            )

        cached = next_cached
        controller.clear_layer_module_cache(layer_idx)

    # D. Final RMSNorm
    if neuron in {"phase", "mtn"}:
        ...
```

---

# 17. 统一 controller 语义判断

所有 statistics 分支统一使用：

```python
controller.collecting_statistics
```

所有 temporal 分支统一使用：

```python
controller.temporal_execution_enabled
```

逐步清理：

```python
controller.mode == "collect"
controller.mode.startswith("deploy_")
```

避免 `calibration_collect` 被漏掉。

---

# 18. Direct block kwargs

`_run()` 应类似：

```python
kwargs = _to_device(item.kwargs, device)
kwargs["use_cache"] = False

cache = fresh_prefix_dynamic_cache(...)

if cache is not None:
    kwargs["past_key_value"] = cache
```

禁止：

```text
past_key_values
```

---

# 19. Validator runtime provenance

当前 neuron constructor 为 validator 使用 `T=1` 做了绕过，不建议继续。

应把 schema validation 与 runtime construction 分离。

建议新增：

```python
validate_phase_state_schema(...)
validate_mtn_state_schema(...)
```

training / conversion / evaluation 有 cfg 时，严格检查：

```text
calibration_phase_T
calibration_mtn_T
calibration_mtn_K
calibration_mtn_threshold_factor
calibration_gif_temporal_steps
```

不要依赖 fake `T=1`。

---

# 20. 必须新增真正端到端 runtime tests

新增：

```text
tests/test_blockwise_calibration_runtime.py
```

必须真实调用：

```python
collect_blockwise_snn_conditioned_statistics(...)
```

不能只测 helper。

至少覆盖：

1. temporal bootstrap shape
2. current block state materialization
3. Block 1 stats 确实来自 Block 0 SNN output
4. Site 3/4 attention-head layout
5. GIF Site 1/3/4/6/7/10 saliency 完整
6. GIF state 可成功 materialize
7. Prefix 使用 `past_key_value`
8. Prefix statistics exclusion
9. tuple/dict Tensor recursive device transfer
10. module cache freshness
11. mixed trajectories
12. runtime manifest dependency
13. all-false regression

---

# 21. 真实 smoke test

代码和 pytest 完成后，用小样本：

```yaml
calibration:
  num_samples: 2
  phase_previous_layers_snn: false
  gif_previous_layers_snn: true
  mtn_previous_layers_snn: false
```

运行：

```bash
python scripts/calibrate_sites.py   --config "$CFG"   --stage ann_training   --calibration-phase A
```

必须完整跑过：

```text
ANN common pass
→ GIF block 0 collect
→ GIF block 0 state materialize
→ GIF block 0 Temporal deploy
→ ...
→ final Stage A manifest
```

---

# 22. Smoke test artifact 检查

随机检查：

```text
layer_000/site_01_*/
layer_001/site_01_*/
```

应存在：

```text
statistics.pt
gif_statistics.pt
phase_state.pt
gif_state.pt
mtn_state.pt
```

且 manifest 中：

```text
gif_state source -> gif_statistics.pt
phase_state source -> statistics.pt
mtn_state source -> statistics.pt
```

---

# 23. Failure policy

以下全部 fail-fast：

- target statistics missing
- target state missing
- wrong site dirname
- malformed Prefix cache
- temporal batch incompatible with T
- Site 3/4 shape mismatch
- GIF saliency role missing
- runtime provenance mismatch
- stale target-specific stats
- state source hash mismatch

不要再用：

```python
continue
```

掩盖问题。

---

# 24. 不在本轮范围内的内容

不要修改：

- GIF 4-bit + 1-bit policy
- GIF high_qmax=30
- GIF T=2
- Phase base=2
- Phase threshold equation
- Phase EMA
- MTN ×2 statistics
- Site topology
- Site 2 all-low
- Site 5 identity
- Site 6 merged-head
- Site 8/9 GIF identity
- common Clip
- Prefix discovery
- rotation
- ANN fine-tuning neuron behavior
- dataset / lm-eval tasks
- learning rate / scheduler
- checkpoint policy

---

# 25. 建议修改文件

至少检查/修改：

```text
scripts/calibrate_sites.py

snn2/blockwise_calibration.py
snn2/controller.py
snn2/model_integration.py
snn2/temporal_model.py
snn2/prefix_cache.py
snn2/calibration.py
snn2/artifacts.py
snn2/state_validation.py
snn2/neurons.py
snn2/config.py
```

测试：

```text
tests/test_blockwise_calibration_runtime.py
tests/test_blockwise_calibration_config.py
tests/test_temporal_ops.py
tests/test_state_validation.py
```

---

# 26. 推荐修复顺序

1. 修 `materialize_target_state` 路径；
2. 修 controller `site_root`；
3. 修 `past_key_value`；
4. 修 temporal bootstrap；
5. 修 recursive kwargs transfer；
6. 修 Site 3/4 collect topology；
7. 修 GIF saliency gate；
8. 修 Prefix statistics exclusion；
9. 修 current-layer module cache；
10. 修 manifest conditional dependency；
11. 修 validator runtime provenance；
12. 补 E2E tests；
13. 跑 `pytest -q`；
14. 跑真实 GIF block-wise Stage A smoke test。

---

# 27. 最终验收标准

- [ ] `gif_previous_layers_snn=true` 可完整完成 Stage A；
- [ ] Phase/MTN true 也可进入完整 block-wise 流程；
- [ ] temporal bootstrap 使用 `[T*B,L]`；
- [ ] embedding temporal policy 正确；
- [ ] Prefix temporal policy 正确；
- [ ] direct layer 使用 `past_key_value`；
- [ ] controller 有正确 `site_root`；
- [ ] target state 写入真实 site dirname；
- [ ] current block deploy 使用刚生成的新 state；
- [ ] Site 3/4 仍为 attention-head statistics；
- [ ] GIF Site 1/3/4/6/7/10 saliency 全部存在；
- [ ] role-specific GIF masks 正确；
- [ ] Prefix statistics domain 与 legacy ANN 一致；
- [ ] tuple Tensor 可跨 GPU 正确迁移；
- [ ] module cache 不使用旧 state；
- [ ] Final RMSNorm Phase/MTN sequential stats 正确；
- [ ] Stage A runtime dependency metadata 正确；
- [ ] training provenance 正确；
- [ ] conversion provenance 正确；
- [ ] all-false legacy path 数值不变；
- [ ] 正式 SNN deployment 数值不变；
- [ ] E2E blockwise runtime test 存在；
- [ ] `pytest -q` 全部通过；
- [ ] `num_samples=2` 的真实 GIF Stage A smoke test 成功。

---

# 28. 最关键执行链

最终代码必须真正实现：

```text
ANN common statistics
        |
        v
GIF target trajectory
        |
        v
temporal bootstrap [T*B,L,D]
        |
        v
Block 0 temporal operators
current Block 0 sites collect/bypass
        |
        v
gif_statistics.pt
        |
        v
gif_state.pt
        |
        v
clear Block 0 module cache
        |
        v
Block 0 real Temporal GIF deploy
        |
        v
X_1^GIF-SNN
        |
        v
Block 1 collect
        |
        v
gif_statistics.pt
        |
        v
gif_state.pt
        |
        v
Block 1 real Temporal GIF deploy
        |
       ...
```

其中：

```text
current block collect
```

永远不会提前运行 current block neuron；

而：

```text
next block input
```

必须来自刚刚完成 calibration 的 current block Temporal SNN output。

这才是本轮要真正完成的 SparseLLM-style block-wise SNN-aware calibration。
