# SAT-LLM Energy Profiler 第五轮修正方案：Empty-Prefix 合法语义与 Prefix Provenance 三态化

> 适用仓库：`https://github.com/wangwk699/SNN`  
> 当前基线：`main` 已完成 v3 最终部署 Energy protocol、mixed temporal counting 修正、Prefix provenance configured/actual 区分、四行 sample identity 检查。  
> 本轮目标：修复 **Prefix policy 已启用但 Prefix discovery 得到空 token 集合** 时 Energy profiler 会错误报错的问题，并把 Prefix provenance 明确拆成“配置态 / 解析策略态 / 实际 KV 注入态”三层。  
> **最高优先级约束：不修改 MAC/AC accounting 数学，不修改 ANN/SNN forward，不修改 Step 9/Step 10 的 conversion/evaluation 语义。**

---

# 1. 当前问题

现有 Energy profiler 中：

```python
actual_enabled = bool(layout.energy_prefix_enabled(neuron))

validate_energy_prefix_runtime(
    actual_enabled=actual_enabled,
    cache=cache,
    prefix_tokens=prefix_tokens,
)
```

而 `validate_energy_prefix_runtime()` 当前会把：

```text
actual_enabled == True
```

理解为：

```text
本次 forward 必须加载一个非空 Prefix KV cache
```

并在：

```python
cache is None
```

时报错。

这在当前项目中不是总成立。

---

# 2. 合法 empty-prefix 场景

Prefix discovery 可以合法得到：

```text
prefix_token_ids = []
```

当前 `snn2/prefix.py` 中：

```text
Qwen:
append_start_token = False
```

因此 Prefix token 集合并不保证至少包含一个 BOS/start token。

当前 `snn2/prefix_cache.py` 又明确：

```python
def build_prefix_key_values(model, prefix_ids):
    if not ids:
        return None
```

所以完全合法的执行链为：

```text
evaluation.prefix_enabled = true
        ↓
resolved Prefix policy = enabled
        ↓
Prefix discovery 得到 prefix_token_ids = []
        ↓
没有 prefixed_key_values.pt
        ↓
prefix_key_values_for_stage(...) 返回 None
        ↓
实际 forward 不注入 Prefix KV cache
```

正式 evaluation 允许这种情况。

Energy profiler 也必须允许。

---

# 3. 当前错误

现有实现会把：

```text
resolved Prefix policy enabled
```

错误等同于：

```text
actual Prefix KV cache injected
```

于是：

```text
policy enabled
cache == None
```

会触发 RuntimeError。

这会让合法 Qwen Energy profiling 失败。

---

# 4. Prefix provenance 必须拆成三层

本轮后明确区分：

```text
1. configured switch
2. resolved Prefix policy
3. actual KV-cache injection
```

不能再让一个 `actual_enabled` 同时承担三种含义。

---

# 5. 三层状态的定义

## 5.1 Configured state

表示 YAML 中用户配置值：

```python
configured_evaluation_prefix_enabled = bool(
    cfg["evaluation"]["prefix_enabled"]
)
```

这是原始配置态。

---

## 5.2 Resolved policy state

表示按照当前 Final ANN / Final SNN 协议，当前 run 是否属于：

```text
Prefix-enabled protocol
```

使用：

```python
resolved_prefix_policy_enabled = bool(
    layout.energy_prefix_enabled(neuron)
)
```

例如：

```text
vanilla + ann
```

即使 YAML：

```yaml
evaluation:
  prefix_enabled: true
```

最终协议仍是：

```text
resolved_prefix_policy_enabled = false
```

因为 Final vanilla ANN 强制不使用 Prefix。

---

## 5.3 Actual runtime injection state

表示本次 forward 是否实际加载了一个非空 KV cache。

定义：

```python
prefix_cache_loaded = cache is not None
actual_evaluation_prefix_enabled = prefix_cache_loaded
```

如果 discovery 得到空 Prefix：

```text
resolved policy = true
cache = None
actual = false
```

这是合法状态。

---

# 6. 推荐 metadata 字段

最终 metadata 至少包含：

```json
{
  "configured_evaluation_prefix_enabled": true,
  "resolved_prefix_policy_enabled": true,
  "prefix_cache_loaded": false,
  "actual_evaluation_prefix_enabled": false,
  "prefix_artifact_stage": "pre_finetuning",
  "prefix_length": 0,
  "energy_path_prefix_enabled": true
}
```

---

# 7. `energy_path_prefix_enabled` 的定义

`energy_path_prefix_enabled` 必须继续表示：

```text
resolved Prefix policy
```

即：

```python
energy_path_prefix_enabled = resolved_prefix_policy_enabled
```

不要改成：

```python
cache is not None
```

原因：

Energy 路径是实验协议 identity，而不是当前 Prefix token 集合是否为空。

所以：

```text
policy enabled + empty Prefix
```

仍应保存在：

```text
prefix_enabled_true
```

路径下。

---

# 8. `actual_evaluation_prefix_enabled` 的定义

重新定义为：

```python
actual_evaluation_prefix_enabled = (cache is not None)
```

表示：

> 本次 Energy forward 是否真正注入了非空 Prefix KV cache。

这是 runtime execution state。

---

# 9. `prefix_cache_loaded` 字段

建议显式新增：

```json
"prefix_cache_loaded": true/false
```

虽然它与：

```text
actual_evaluation_prefix_enabled
```

当前语义一致，但保留该字段能让 metadata 更自解释：

```text
actual_evaluation_prefix_enabled
= 是否真正用了 Prefix

prefix_cache_loaded
= 是否实际加载了 KV cache
```

---

# 10. `prefix_artifact_stage` 的正确语义

本轮建议：

```python
if resolved_prefix_policy_enabled:
    prefix_artifact_stage = (
        final_ann_evaluation_prefix_artifact_stage(cfg)
        if neuron == "ann"
        else final_snn_evaluation_prefix_artifact_stage(cfg)
    )
else:
    prefix_artifact_stage = None
```

即：

```text
artifact stage 跟随 resolved policy
```

而不是跟随 cache 是否为空。

原因：

即使 Prefix token 集为空，当前 run 仍然读取/依赖对应 stage 的：

```text
prefix_state.json
```

来确定“Prefix 为空”。

所以 empty-prefix 时 stage 仍然有意义。

---

# 11. 三类典型 metadata

## 11.1 Vanilla ANN

```text
configured=true
resolved_policy=false
cache_loaded=false
actual=false
artifact_stage=null
prefix_length=0
energy_path_prefix_enabled=false
```

示例：

```json
{
  "configured_evaluation_prefix_enabled": true,
  "resolved_prefix_policy_enabled": false,
  "prefix_cache_loaded": false,
  "actual_evaluation_prefix_enabled": false,
  "prefix_artifact_stage": null,
  "prefix_length": 0,
  "energy_path_prefix_enabled": false
}
```

---

## 11.2 Aware SNN + 非空 Prefix

```text
configured=true
resolved_policy=true
cache_loaded=true
actual=true
artifact_stage=pre_finetuning 或 post_finetuning
prefix_length>0
energy_path_prefix_enabled=true
```

---

## 11.3 Aware SNN + empty Prefix

```text
configured=true
resolved_policy=true
cache_loaded=false
actual=false
artifact_stage=pre_finetuning 或 post_finetuning
prefix_length=0
energy_path_prefix_enabled=true
```

这是本轮新增必须支持的合法状态。

---

# 12. Runtime consistency check 必须放宽

当前错误规则：

```python
if actual_enabled and cache is None:
    raise RuntimeError(...)
```

必须删除。

不能再把：

```text
policy enabled
```

强制映射成：

```text
cache 必须存在
```

---

# 13. 正确的 runtime consistency check

建议改成只检查：

```text
cache state
与
prefix_length
```

的一致性。

推荐：

```python
def validate_energy_prefix_runtime(
    *,
    cache: Any | None,
    prefix_tokens: int,
) -> None:
    cache_loaded = cache is not None

    if cache_loaded and prefix_tokens <= 0:
        raise RuntimeError(
            "Loaded Prefix KV cache must have positive Prefix length"
        )

    if not cache_loaded and prefix_tokens != 0:
        raise RuntimeError(
            "Prefix length must be zero when no Prefix KV cache is loaded"
        )
```

这两个规则足够。

---

# 14. 不再把 policy state 传给 runtime validator

推荐删除参数：

```python
actual_enabled
```

或：

```python
resolved_prefix_policy_enabled
```

因为 runtime validator 不负责判断 policy。

其职责只应是：

```text
cache 与 length 自洽
```

---

# 15. `energy_prefix_provenance()` 推荐实现

建议重构为：

```python
def energy_prefix_provenance(
    cfg,
    layout,
    *,
    neuron,
    cache,
    prefix_tokens,
):
    configured = bool(cfg["evaluation"]["prefix_enabled"])
    resolved_policy = bool(layout.energy_prefix_enabled(neuron))
    cache_loaded = cache is not None

    validate_energy_prefix_runtime(
        cache=cache,
        prefix_tokens=prefix_tokens,
    )

    actual = cache_loaded

    if resolved_policy:
        artifact_stage = (
            final_ann_evaluation_prefix_artifact_stage(cfg)
            if neuron == "ann"
            else final_snn_evaluation_prefix_artifact_stage(cfg)
        )
    else:
        artifact_stage = None

    return {
        "configured_evaluation_prefix_enabled": configured,
        "resolved_prefix_policy_enabled": resolved_policy,
        "prefix_cache_loaded": cache_loaded,
        "actual_evaluation_prefix_enabled": actual,
        "prefix_artifact_stage": artifact_stage,
        "prefix_length": prefix_tokens,
        "energy_path_prefix_enabled": resolved_policy,
    }
```

---

# 16. 建议增加一个 policy/cache sanity check

虽然：

```text
resolved_policy=true + cache=None
```

是合法的 empty-prefix 情况，

但是：

```text
resolved_policy=false + cache!=None
```

理论上不应该发生。

因此可以额外检查：

```python
if not resolved_policy and cache_loaded:
    raise RuntimeError(
        "Prefix KV cache was loaded although the resolved Energy Prefix policy is disabled"
    )
```

这是安全的。

---

# 17. 不要反向禁止 empty Prefix

禁止写：

```python
if resolved_policy and not cache_loaded:
    raise
```

这正是本轮必须删除的错误。

---

# 18. `prefix_key_values_for_stage()` 不修改

当前正式 modeling 逻辑已经允许：

```text
empty prefix -> None
```

不要修改：

```text
snn2/modeling.py
snn2/prefix_cache.py
snn2/prefix.py
```

本轮只修 Energy profiler 的 provenance/validation。

---

# 19. 不修改 Prefix discovery

不要人为强制：

```text
Qwen Prefix 至少 1 token
```

也不要为了 Energy profiler 给空 Prefix 自动追加 BOS。

这会改变模型实际 protocol，禁止。

---

# 20. 测试：empty Prefix 必须合法

新增：

```python
def test_resolved_enabled_empty_prefix_is_valid():
    ...
```

构造：

```text
configured=true
resolved_policy=true
cache=None
prefix_tokens=0
```

预期：

```text
不抛异常
```

metadata：

```text
resolved_prefix_policy_enabled == True
prefix_cache_loaded == False
actual_evaluation_prefix_enabled == False
prefix_length == 0
energy_path_prefix_enabled == True
prefix_artifact_stage != None
```

---

# 21. 测试：normal Prefix

保留/更新：

```text
configured=true
resolved=true
cache!=None
prefix_tokens=16
```

预期：

```text
cache_loaded=true
actual=true
```

---

# 22. 测试：vanilla ANN

保留：

```text
configured=true
resolved=false
cache=None
prefix_tokens=0
```

预期：

```text
actual=false
artifact_stage=None
energy_path_prefix_enabled=false
```

---

# 23. 测试：disabled policy + cache loaded 必须报错

新增：

```text
resolved=false
cache!=None
prefix_tokens>0
```

预期：

```text
RuntimeError
```

说明真正发生了协议与执行不一致。

---

# 24. 测试：cache/length inconsistency

继续测试：

```text
cache=None, prefix_tokens>0 -> raise
cache!=None, prefix_tokens<=0 -> raise
```

---

# 25. 更新现有 prefix runtime tests

当前测试中类似：

```python
(True, None, 0) -> raise
```

必须改掉。

新的正确预期：

```text
policy enabled + no cache + length 0
```

是合法。

因此不要继续把它列在 invalid cases。

---

# 26. 更新 metadata schema version

本轮 metadata schema 再发生语义变化。

建议：

```python
ENERGY_METADATA_SCHEMA_VERSION = 3
```

保持：

```python
ENERGY_PROFILER_VERSION = 3
ENERGY_ACCOUNTING_POLICY = "sat_llm_mac_ac_v3_final_deployment_protocol"
```

不变。

原因：

```text
accounting 数学和 deployment protocol 没变；
只是 Prefix metadata schema 更精确。
```

---

# 27. 不升级 accounting policy

不要改：

```python
ENERGY_ACCOUNTING_POLICY
```

因为：

```text
MAC/AC 数学无变化
```

---

# 28. 更新 `实验执行总结.md`

在 Step 11 的 Prefix provenance 小节中改成三层解释：

```text
configured_evaluation_prefix_enabled
resolved_prefix_policy_enabled
actual_evaluation_prefix_enabled
```

并增加：

> 当 resolved Prefix policy 为 true，但 Prefix discovery 得到空 `prefix_token_ids` 时，可能不存在 KV cache。此时该 run 仍属于 Prefix-enabled protocol/path，但本次 forward 实际不注入 Prefix，因此 `prefix_cache_loaded=false`、`actual_evaluation_prefix_enabled=false`、`prefix_length=0`；`prefix_artifact_stage` 仍记录被解析的 Pre/Post Prefix stage。

---

# 29. 更新 `代码结构总结.md`

将 `energy_profiler.py` 描述可更新为：

```text
校验最终部署来源、区分 Prefix configured/resolved/actual 三态，并用临时 hook 统计固定 held-out 序列理论能耗。
```

---

# 30. 不修改 sample identity 规则

上一轮新增的四行检查继续保持：

```text
validation_manifest_sha256
selected_validation_positions
profile_num_samples
profile_seed
profile_sequence_length
```

全部一致。

本轮不改。

---

# 31. 不修改 selected-aware source 规则

继续保持：

```text
ANN -> vanilla
SNN -> 同一个 selected phase_aware/gif_aware checkpoint
```

不改。

---

# 32. 不修改 Energy counting

保持：

```text
4.6 pJ/MAC
0.9 pJ/AC
```

以及：

```text
dynamic × dynamic -> MAC
event × event -> AC
mixed dynamic/event -> active MAC
```

全部不改。

---

# 33. 不修改输出路径结构

当前：

```text
energy/ann/<prefix-policy-path>/profile_num_samples_<N>/
energy/snn/.../<prefix-policy-path>/profile_num_samples_<N>/
```

保持。

路径里的 Prefix bool 表示：

```text
resolved policy
```

不是：

```text
cache 是否非空
```

---

# 34. 旧 metadata 的兼容说明

旧 schema v2：

```text
actual_evaluation_prefix_enabled
```

实际上等同于：

```text
resolved policy
```

本轮 schema v3 后：

```text
actual_evaluation_prefix_enabled
```

明确表示：

```text
actual KV cache injection
```

因此正式论文结果建议重新跑一次，避免 provenance 语义混淆。

---

# 35. 推荐修改文件

主要：

```text
snn2/energy_profiler.py
tests/test_energy_profiler.py
实验执行总结.md
代码结构总结.md
```

通常不需要修改：

```text
scripts/profile_energy.py
```

---

# 36. 不修改以下文件

不要修改：

```text
snn2/prefix.py
snn2/prefix_cache.py
snn2/modeling.py
snn2/evaluation.py
snn2/controller.py
snn2/temporal_model.py
snn2/temporal_ops.py
snn2/neurons.py
scripts/convert_snn.py
scripts/evaluate_tldr.py
scripts/evaluate_lm_harness.py
```

除非仅为 import/type 引用。

---

# 37. 回归测试

修改后运行：

```bash
python scripts/materialize_configs.py \
  --matrix configs/experiment_matrix.yaml \
  --output-dir configs/generated

pytest -q
```

全部通过。

---

# 38. 最终验收标准

### Empty Prefix

- [ ] resolved policy=true + cache=None + length=0 合法；
- [ ] 不再因为 empty Prefix 报错；
- [ ] Qwen 空 Prefix 情况可正常 profiling。

### Metadata

- [ ] configured state 单独记录；
- [ ] resolved policy 单独记录；
- [ ] cache_loaded 单独记录；
- [ ] actual state 表示实际 KV 注入；
- [ ] path state 表示 resolved policy；
- [ ] empty Prefix 时 artifact stage 仍保留 Pre/Post stage。

### Runtime safety

- [ ] cache=None + length>0 报错；
- [ ] cache!=None + length<=0 报错；
- [ ] resolved policy=false + cache!=None 报错。

### No regression

- [ ] v3 deployment source protocol不变；
- [ ] mixed MAC/AC counting不变；
- [ ] sample identity规则不变；
- [ ] Step 9/10不变；
- [ ] pytest全部通过。

---

# 39. Codex 完成后需汇报

完成后返回：

```text
1. 修改文件
2. Prefix configured/resolved/actual 三层定义
3. empty-prefix 的合法处理方式
4. prefix_artifact_stage 新语义
5. ENERGY_METADATA_SCHEMA_VERSION
6. 修改/新增测试
7. pytest -q 结果
8. vanilla ANN metadata 示例
9. aware SNN non-empty Prefix metadata 示例
10. aware SNN empty Prefix metadata 示例
```

不要修改当前 Energy accounting 数学和最终 deployment source protocol。
