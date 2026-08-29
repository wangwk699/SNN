# SNN 项目补充修正方案：verify_artifacts Topology Validation + Saliency Role/Provenance Validation 加强

> 目标仓库：`https://github.com/wangwk699/SNN`
>
> 目标分支：`main`
>
> 本文档面向部署在服务器上的 Codex。假设 Codex 没有本次对话上下文，必须仅凭本文档完成代码修改。
>
> 本轮只做以下 3 项补充修正：
>
> 1. 修正 `scripts/verify_artifacts.py` 中已经过时的 Site 2/3/4/6 topology validation；
> 2. 加强 `snn2/state_validation.py` 对 saliency role map 的 key 集合校验；
> 3. 加强 `scripts/verify_artifacts.py` 对 evaluation GIF provenance metadata 的校验。
>
> 不允许修改上一轮已经完成并通过测试的核心 forward、calibration、GIF/Phase/MTN 算法和 topology。

---

# 1. 当前最终 topology 作为本轮唯一真值

当前项目已经完成以下 topology 重构，本轮所有 validation 必须以此为准。

## Site 2

Site 2 仍是 per-head：

```text
layout_kind = attention_head
parameter_layout = attention_head_grouped
num_heads = num_attention_heads
channels_per_head = head_dim
```

GIF 为：

```text
all-low static 4-bit
```

---

## Site 3 / Site 4

Site 3/4 已经移动到：

```text
repeat_kv
```

之后。

因此 calibration statistics 使用的是 repeated attention heads：

```text
[B, H_attn, L, D]
```

实际 ANN replacement tensor是：

```text
[B, L, H_attn*D]
```

但 parameter/statistics logical layout仍为：

```text
attention_head
attention_head_grouped
```

关键：

```text
num_heads = num_attention_heads
```

不是：

```text
num_key_value_heads
```

因此对于 GQA 模型：

```text
H_attn != H_kv
```

Site 3/4 必须校验为 query/attention head 数。

---

## Site 6

Site 6 已经改为 head merge 后 replacement：

```text
[B,H,L,D]
 -> [B,L,H,D]
 -> [B,L,HD]
 -> Site 6
```

所以 Site 6 不再 per-head。

正确 statistics/state layout：

```text
layout_kind = last_dim
parameter_layout = last_dim_grouped
num_heads = None
channels = hidden_size
```

其中：

```text
hidden_size = num_attention_heads * head_dim
```

Site 6 只保留：

```text
group_size
```

不允许再用 query heads / KV heads 解释 Site 6 state。

---

# 2. 修正 `scripts/verify_artifacts.py::_verify_grouped_calibration()`

当前代码仍保留旧 topology 判断，大致如下：

```python
if site in {2, 6} and int(statistics["num_heads"]) != query_heads:
    raise ...

if site in {3, 4} and int(statistics["num_heads"]) != kv_heads:
    raise ...
```

这是错误的。

其中：

- Site 3/4 已经不是 native KV-head coordinate；
- Site 6 的 `num_heads` 正确值是 `None`，`int(None)` 会直接触发 `TypeError`。

必须整体替换。

---

# 3. `verify_artifacts.py` 中正确的 Site 2/3/4 validation

文件：

```text
scripts/verify_artifacts.py
```

函数：

```python
_verify_grouped_calibration(...)
```

在读取：

```python
ann_config = read_json(layout.ann_checkpoint_dir / "config.json")
query_heads = int(ann_config["num_attention_heads"])
kv_heads = int(ann_config.get("num_key_value_heads", query_heads))
```

后：

```text
kv_heads
```

仍可保留供其他检查使用，但 Site 3/4 不再使用它。

推荐同时读取：

```python
hidden_size = int(ann_config["hidden_size"])
```

若模型 config 显式存在：

```text
head_dim
```

可额外读取：

```python
head_dim = int(
    ann_config.get("head_dim", hidden_size // query_heads)
)
```

并检查：

```python
query_heads * head_dim == hidden_size
```

如果不成立，报：

```text
ANN config attention-head geometry is inconsistent
```

但如果当前支持的模型 config 中 `head_dim` 可能不是必填，保持 fallback 即可。

---

## 3.1 Site 2/3/4 必须统一校验为 attention-head statistics

建议：

```python
if site in {2, 3, 4}:
    if statistics.get("layout_kind") != "attention_head":
        raise ValueError(
            f"Site {site} must use attention_head statistics: {statistics_path}"
        )

    if not isinstance(statistics.get("num_heads"), int):
        raise ValueError(
            f"Site {site} must record integer num_heads: {statistics_path}"
        )

    if int(statistics["num_heads"]) != query_heads:
        raise ValueError(
            f"Site {site} must use repeated/query attention heads "
            f"({query_heads}), got {statistics['num_heads']}: {statistics_path}"
        )

    channels_per_head = statistics.get("channels_per_head")
    if not isinstance(channels_per_head, int) or channels_per_head <= 0:
        raise ValueError(
            f"Site {site} has invalid channels_per_head: {statistics_path}"
        )

    expected_channels = query_heads * channels_per_head
    if int(statistics.get("channels", -1)) != expected_channels:
        raise ValueError(
            f"Site {site} channel count must equal "
            f"num_attention_heads * channels_per_head: {statistics_path}"
        )
```

重点：

```text
Site 2/3/4 全部 num_heads == num_attention_heads
```

---

# 4. Site 6 必须独立校验为 merged last-dim

不要再写：

```python
int(statistics["num_heads"])
```

Site 6 正确 validation：

```python
if site == 6:
    if statistics.get("layout_kind") != "last_dim":
        raise ValueError(
            f"Site 6 must use merged last_dim statistics: {statistics_path}"
        )

    if statistics.get("num_heads") is not None:
        raise ValueError(
            f"Site 6 must not preserve a per-head layout: {statistics_path}"
        )

    if statistics.get("channels_per_head") is not None:
        raise ValueError(
            f"Site 6 must not save channels_per_head: {statistics_path}"
        )

    if int(statistics.get("channels", -1)) != hidden_size:
        raise ValueError(
            f"Site 6 merged width must equal hidden_size={hidden_size}: "
            f"{statistics_path}"
        )
```

如果当前 `statistics.pt` 对 `last_dim` layout 的 `channels_per_head` 不是 `None` 而是其他已有 schema 值，则先检查 `snn2/stats.py` 的真实 schema，不要凭本文档硬改 artifact schema。

唯一不可变语义是：

```text
Site 6:
layout_kind = last_dim
num_heads = None
channels = hidden_size
```

如果 `channels_per_head` 对 last-dim schema 被项目统一用于记录 width，那么不要额外禁止；以现有 `stats.py` schema 为准。

因此 Codex 实施时优先检查当前 `SiteStatistics.state_dict()`。

---

# 5. Site 3/4 不再校验 native KV heads

必须删除所有类似：

```python
if site in {3, 4} and int(statistics["num_heads"]) != kv_heads:
    raise ValueError(
        f"Site {site} must use native KV heads"
    )
```

当前正确含义改为：

```text
Site 3/4 use post-repeat attention heads
```

错误信息也必须同步更新。

建议统一错误文本：

```text
Site 3/4 must use post-repeat attention-head geometry
```

不要再出现：

```text
native KV heads
```

---

# 6. 同时校验 materialized state，而不只 statistics

为了让 `verify_artifacts.py` 真正验证最终 bundle，建议在 `_verify_grouped_calibration()` 内，在加载 `statistics.pt` 后继续加载：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt（若存在）
```

至少对 Site 2/3/4/6 检查 parameter layout。

---

## 6.1 Site 2/3/4 state

对于：

```text
phase
gif
mtn
clip（若存在）
```

要求：

```text
parameter_layout == attention_head_grouped
num_heads == num_attention_heads
```

如果项目还支持：

```text
attention_head_scalar
```

用于 `group_size=-1` 的旧特殊表示，则以当前 `build_phase_state()` / `_layout_from_statistics()` 生成结果为准。

本轮不应改变已有 state schema。

关键必须成立：

```text
Site 2/3/4 的 logical head 数 = num_attention_heads
```

---

## 6.2 Site 6 state

对 Site 6：

```text
phase_state.pt
gif_state.pt
mtn_state.pt
clip_state.pt（若存在）
```

要求：

```text
parameter_layout == last_dim_grouped
num_heads is None
```

并且 parameter width必须对应 hidden_size/group_size。

如果 generic `state_validation.py` 已经检查，这里可以只做 minimum topology sanity check，避免重复大量逻辑。

---

# 7. 修正点 2：加强 saliency role map key validation

文件：

```text
snn2/state_validation.py
```

当前 salient GIF validation 主要有：

```python
saved_rules = state.get("saliency_rule_by_role")
if (
    not isinstance(saved_rules, dict)
    or set(saved_rules.values()) != {expected_rule}
):
    raise ...

saved_dtypes = state.get("saliency_accumulator_dtype_by_role")
if (
    not isinstance(saved_dtypes, dict)
    or set(saved_dtypes.values()) != {expected_dtype}
):
    raise ...
```

这只校验 value，没有严格校验 role key。

例如理论异常：

```python
Site 1:
saliency_rule_by_role = {
    "q": "spikellm_linear_fp32"
}
```

value 集仍为：

```text
{"spikellm_linear_fp32"}
```

因此 role completeness 没有被严格表达。

---

# 8. 定义每个 salient site 的 expected roles

建议使用现有：

```python
GIF_MULTI_MASK_ROLES
```

构造：

```python
expected_roles = GIF_MULTI_MASK_ROLES.get(
    site_index,
    ("default",),
)
```

则：

```text
Site 1  -> ("q","k","v")
Site 7  -> ("gate","up")
Site 3  -> ("default",)
Site 4  -> ("default",)
Site 6  -> ("default",)
Site 10 -> ("default",)
```

---

# 9. `saliency_rule_by_role` 必须同时校验 keys 和 values

改成：

```python
saved_rules = state.get("saliency_rule_by_role")

if (
    not isinstance(saved_rules, dict)
    or tuple(saved_rules.keys()) != tuple(expected_roles)
    or any(saved_rules[role] != expected_rule for role in expected_roles)
):
    raise ValueError(
        f"Invalid GIF saliency rule/role map at {state_path}"
    )
```

是否要求字典顺序完全等于 expected roles，取决于当前 artifact 写入是否保证 insertion order。

当前 build logic 是按：

```python
expected_roles
```

循环构建，因此可以严格要求顺序。

如果希望只验证集合，则：

```python
set(saved_rules) == set(expected_roles)
```

即可。

推荐：

```text
key set必须严格一致
values逐 role严格一致
```

顺序通过 manifest 独立验证。

例如：

```python
if set(saved_rules) != set(expected_roles):
    ...
for role in expected_roles:
    if saved_rules.get(role) != expected_rule:
        ...
```

---

# 10. dtype role map 同样加强

改成：

```python
saved_dtypes = state.get(
    "saliency_accumulator_dtype_by_role"
)

if (
    not isinstance(saved_dtypes, dict)
    or set(saved_dtypes) != set(expected_roles)
):
    raise ValueError(...)

for role in expected_roles:
    if saved_dtypes.get(role) != expected_dtype:
        raise ValueError(...)
```

---

# 11. multi-role mask 本身也建议严格 key 对齐

当前对 Site 1/7 已检查：

```python
tuple(state.get("mask_roles", ())) == roles
```

建议再确认：

```python
mask_low_by_role
saliency_score_by_role
```

的 key 与：

```python
expected_roles
```

完全一致。

例如：

```python
if site_index in GIF_MULTI_MASK_ROLES:
    masks = state.get("mask_low_by_role")
    scores = state.get("saliency_score_by_role")

    if not isinstance(masks, dict) or set(masks) != set(expected_roles):
        raise ValueError(...)

    if not isinstance(scores, dict) or set(scores) != set(expected_roles):
        raise ValueError(...)
```

这不会改变 forward，只是加强 corrupted artifact fail-fast。

---

# 12. single-mask site 必须只有 default role

对：

```text
Site 3
Site 4
Site 6
Site 10
```

要求：

```python
expected_roles == ("default",)
```

并且：

```text
saliency_rule_by_role.keys() == {"default"}
saliency_accumulator_dtype_by_role.keys() == {"default"}
```

不要允许额外 role。

---

# 13. 修正点 3：`verify_artifacts.py` 校验 evaluation GIF provenance metadata

当前：

```text
evaluation_forward_metadata()
```

已经输出：

```text
gif_salient_site_ids
gif_all_low_site_ids
gif_identity_site_ids
gif_multi_mask_roles
gif_saliency_selection_policy
gif_saliency_tie_policy
gif_linear_saliency_dtype
gif_matmul_saliency_dtype
```

但：

```python
_verify_final_ann_forward_metadata(...)
```

只校验了：

```text
static_replacement_impl
```

没有验证这些新增 provenance。

本轮顺手补全。

---

# 14. `verify_artifacts.py` 增加 imports

从：

```text
snn2.sites
```

增加：

```python
GIF_ALL_LOW_SITE_IDS
GIF_IDENTITY_SITE_IDS
GIF_MULTI_MASK_ROLES
GIF_SALIENT_SITE_IDS
```

从：

```text
snn2.temporal_ops
```

增加：

```python
GIF_LINEAR_SALIENCY_DTYPE
GIF_MATMUL_SALIENCY_DTYPE
GIF_SALIENCY_SELECTION_POLICY
GIF_SALIENCY_TIE_POLICY
```

不要手写字符串常量。

---

# 15. `_verify_final_ann_forward_metadata()` 中加入 GIF provenance

在：

```python
required = {
    ...
}
```

后，再构造：

```python
gif_provenance = {
    "gif_salient_site_ids": sorted(GIF_SALIENT_SITE_IDS),
    "gif_all_low_site_ids": sorted(GIF_ALL_LOW_SITE_IDS),
    "gif_identity_site_ids": sorted(GIF_IDENTITY_SITE_IDS),
    "gif_multi_mask_roles": {
        str(site): list(roles)
        for site, roles in sorted(GIF_MULTI_MASK_ROLES.items())
    },
    "gif_saliency_selection_policy": GIF_SALIENCY_SELECTION_POLICY,
    "gif_saliency_tie_policy": GIF_SALIENCY_TIE_POLICY,
    "gif_linear_saliency_dtype": GIF_LINEAR_SALIENCY_DTYPE,
    "gif_matmul_saliency_dtype": GIF_MATMUL_SALIENCY_DTYPE,
}
```

然后严格逐字段校验：

```python
for key, value in gif_provenance.items():
    if metadata.get(key) != value:
        raise ValueError(
            f"Final ANN evaluation has stale/incompatible GIF provenance "
            f"{key}: {path}. Re-run final ANN evaluation."
        )
```

---

# 16. 是否只在 GIF-aware ANN 校验？

推荐：

```text
所有 final ANN evaluation 都校验这些全局 GIF implementation provenance 字段
```

原因是：

```text
evaluation_forward_metadata()
```

当前已经对：

```text
vanilla
unaware
phase_aware
gif_aware
```

统一写入这些字段。

这些字段描述的是当前代码版本中 GIF implementation policy，而不是“本次 forward 是否实际启用了 GIF”。

因此：

```text
所有 final ANN metadata 都应该携带并验证
```

这样旧 evaluation 文件会统一 fail-fast。

---

# 17. SNN evaluation metadata 也建议继续走 `validate_temporal_policy()`

当前 SNN metrics verification 已执行：

```python
validate_temporal_policy(policy_source, context=...)
```

而：

```text
gif_saliency_selection_policy
gif_saliency_tie_policy
gif_linear_saliency_dtype
gif_matmul_saliency_dtype
```

已经进入：

```python
temporal_policy_metadata()
```

因此 SNN 路径不用再重复手工校验。

保持现状即可。

---

# 18. `verify_artifacts.py::_verify_grouped_calibration()` 顺手校验 manifest topology provenance

在函数开头目前已经通过：

```python
expected = {
    ...
}
_require_manifest_flags(...)
```

建议增加：

```python
"gif_salient_site_ids": sorted(GIF_SALIENT_SITE_IDS),
"gif_all_low_site_ids": sorted(GIF_ALL_LOW_SITE_IDS),
"gif_identity_site_ids": sorted(GIF_IDENTITY_SITE_IDS),
"gif_multi_mask_roles": {
    str(site): list(roles)
    for site, roles in sorted(GIF_MULTI_MASK_ROLES.items())
},
```

因为 `calibration_state_manifest.json` 顶层已经通过：

```python
**topology_metadata()
```

写入这些字段。

这样 `verify_artifacts.py` 会同时检查：

```text
calibration topology provenance
evaluation topology provenance
```

---

# 19. 必须新增 / 更新测试

本轮虽然只是 validation 修正，也必须补测试，避免未来再次回退到 native KV-head topology。

---

## 19.1 给 `verify_artifacts.py::_verify_grouped_calibration()` 增加直接单测

如果当前 tests 没有方便 import 私有 helper，可新增：

```text
tests/test_verify_artifacts.py
```

或在已有 artifact verification tests 中测试。

至少覆盖：

### Case A：GQA 下 Site 3/4 使用 attention heads

构造：

```text
num_attention_heads = 8
num_key_value_heads = 2
```

Site 3/4 statistics：

```text
num_heads = 8
layout_kind = attention_head
```

验证：

```text
通过
```

### Case B：旧 native KV-head state 被拒绝

Site 3：

```text
num_heads = 2
```

必须报错。

### Case C：Site 6 merged last-dim

Site 6：

```text
layout_kind = last_dim
num_heads = None
channels = hidden_size
```

必须通过。

### Case D：Site 6 旧 per-head state

Site 6：

```text
layout_kind = attention_head
num_heads = num_attention_heads
```

必须被拒绝。

### Case E：Site 6 width错误

```text
channels != hidden_size
```

必须被拒绝。

---

# 20. Saliency role validation tests

在：

```text
tests/test_controller_state_loading.py
```

或：

```text
tests/test_calibration_topology.py
```

加入 corrupted artifact tests。

---

## Case A：Site 1 rules 缺 role

例如：

```python
state["saliency_rule_by_role"] = {
    "q": "spikellm_linear_fp32",
    "k": "spikellm_linear_fp32",
}
```

缺少：

```text
v
```

必须 fail。

---

## Case B：Site 1 dtype map 缺 role

```python
state["saliency_accumulator_dtype_by_role"] = {
    "q": "float32",
    "k": "float32",
}
```

必须 fail。

---

## Case C：Site 7 多余 role

```python
{
    "gate": "float32",
    "up": "float32",
    "extra": "float32",
}
```

必须 fail。

---

## Case D：single-mask site role错误

例如 Site 6：

```python
saliency_rule_by_role = {
    "o": "spikellm_linear_fp32"
}
```

必须 fail。

正确必须是：

```text
default
```

---

## Case E：mask_low_by_role 缺 key

Site 1：

```text
mask_roles = [q,k,v]
```

但：

```text
mask_low_by_role 只有 q,k
```

必须 fail。

---

# 21. Evaluation provenance validation tests

对：

```python
_verify_final_ann_forward_metadata(...)
```

至少增加：

### Case A：合法 GIF provenance

完整 metadata 通过。

### Case B：salient site IDs 过时

例如：

```python
gif_salient_site_ids = [1,3,4,7,10]
```

缺少 Site 6，必须 fail。

### Case C：multi-mask role错误

例如：

```python
"1": ["q","k"]
```

必须 fail。

### Case D：dtype错误

```python
gif_matmul_saliency_dtype = "float32"
```

必须 fail。

### Case E：threshold policy错误

```python
gif_saliency_selection_policy = "per_head_topk"
```

必须 fail。

---

# 22. 不需要 bump 任何 format/version

本轮不修改：

- Site topology；
- calibration state schema；
- statistics schema；
- temporal implementation；
- conversion metadata schema；
- calibration manifest schema。

只是：

```text
validator correctness
validator strictness
```

因此不要 bump：

```text
SITE_TOPOLOGY_VERSION
SITE_STATE_FORMAT_VERSION
STATISTICS_FORMAT_VERSION
CALIBRATION_MANIFEST_FORMAT_VERSION
CONVERSION_METADATA_FORMAT_VERSION
TEMPORAL_IMPLEMENTATION_VERSION
```

---

# 23. 不要修改的核心代码

本轮严禁修改：

```text
snn2/model_integration.py
snn2/temporal_model.py
```

除非测试暴露出与 validation 完全无关的明确 bug。

尤其不要改变：

- Site 3/4 post-repeat topology；
- Site 6 merged topology；
- Site 2 all-low quantization；
- Site 1/7 multi-role GIF；
- Site 5/8/9 identity；
- saliency 数值公式；
- FP32/FP64 precision；
- common Clip 数值规则；
- group_size；
- Prefix；
- rotation；
- ANN training；
- SNN deployment。

---

# 24. 建议修改文件

最低：

```text
scripts/verify_artifacts.py
snn2/state_validation.py
```

测试文件建议：

```text
tests/test_calibration_topology.py
tests/test_controller_state_loading.py
tests/test_evaluation_paths.py
```

如果已有专门的 artifact verifier test，则优先放在那里。

---

# 25. 最终 acceptance criteria

## verify_artifacts topology

- [ ] Site 2/3/4 都要求 `layout_kind=attention_head`。
- [ ] Site 2/3/4 都要求 `num_heads=num_attention_heads`。
- [ ] Site 3/4 不再使用 `num_key_value_heads` validation。
- [ ] Site 6 要求 `layout_kind=last_dim`。
- [ ] Site 6 要求 `num_heads=None`。
- [ ] Site 6 `channels=hidden_size`。
- [ ] Site 6 不再执行 `int(None)`。
- [ ] 旧 native-KV Site 3/4 artifact 会 fail-fast。
- [ ] 旧 per-head Site 6 artifact 会 fail-fast。

## saliency role validation

- [ ] Site 1 rules/dtypes keys严格为 `q,k,v`。
- [ ] Site 7 rules/dtypes keys严格为 `gate,up`。
- [ ] Site 3/4/6/10 keys严格为 `default`。
- [ ] `mask_low_by_role` / `saliency_score_by_role` keys 与 multi-role policy一致。
- [ ] 缺 role / 多 role 都 fail-fast。

## evaluation provenance

- [ ] `verify_artifacts.py` 校验 `gif_salient_site_ids`。
- [ ] 校验 `gif_all_low_site_ids`。
- [ ] 校验 `gif_identity_site_ids`。
- [ ] 校验 `gif_multi_mask_roles`。
- [ ] 校验 saliency selection policy。
- [ ] 校验 tie policy。
- [ ] 校验 linear FP32。
- [ ] 校验 matmul FP64。
- [ ] stale evaluation metadata 会要求重新 evaluation。

## versioning

- [ ] 本轮没有不必要的 version bump。

## tests

完成后必须实际运行：

```bash
pytest -q
```

并且只有实际通过后才能报告通过。

若项目已有针对 `verify_artifacts.py` 的 CLI smoke test，也建议额外运行。

---

# 26. Codex 完成后应返回

Codex 应简要说明：

```text
1. 修改了哪些文件；
2. verify_artifacts 中 Site 2/3/4/6 topology validation 如何变化；
3. saliency role key validation 加强了哪些条件；
4. evaluation provenance 新增校验了哪些字段；
5. 是否有 version bump（预期：没有）；
6. pytest -q 的真实结果。
```
