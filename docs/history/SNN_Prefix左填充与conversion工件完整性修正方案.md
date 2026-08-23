# SNN Prefix 左填充位置语义与 conversion 工件完整性修正方案

> 本文档是交给服务器端 Codex 的独立实施说明。服务器端 Codex 不应依赖任何聊天上下文，只需读取本文档和目标仓库即可完成修改、测试与交付。

---

# 0. 基线、目标与硬性约束

## 0.1 代码基线

目标仓库：

```text
https://github.com/wangwk699/SNN
```

本文档基于以下 `main` 提交审查并编写：

```text
47346ceaa8afe9f75668bfe6b5241a01c73430d5
```

服务器端开始修改前必须先执行：

```bash
git status --short
git rev-parse HEAD
git log -5 --oneline
```

要求：

1. 确认当前工作分支确实包含 `47346ce` 的 temporal v2 实现，或者其后续等价提交。
2. 保留用户已有、与本任务无关的工作区修改。
3. 不得使用 `git reset --hard`、`git checkout -- .` 等破坏性命令。
4. 如果目标代码已经发生较大漂移，先按本文的行为约束重新定位函数，不得机械套用旧行号。

## 0.2 本次必须解决的问题

本次只解决以下四类问题：

```text
A. Prefix=true + 左填充 batch 时，没有逐样本 position_ids，
   导致相同样本的结果依赖同 batch 中其它样本的长度。

B. conversion metadata 虽然保存 rotation_state_sha256，
   但 SNN evaluation 和 verify_artifacts 没有验证该 hash。

C. calibration bundle 只检查“已存在层的 10 个 site”，
   没有验证覆盖模型全部 Transformer 层且层编号连续。

D. configs/generated 被 gitignore，但测试依赖预生成的 12 个文件，
   导致干净 clone 直接 pytest 失败；部分路径测试也不跨平台。
```

## 0.3 本次明确不修改的内容

以下内容不得借本任务擅自改动：

1. Phase neuron 的编码公式、`T/base/max_spikes/tau/v0` 方案。
2. GIF 的静态 scalar calibration 方案。
3. MTN 的单一 scalar scale 方案。
4. GIF high qmax=`30`。
5. GIF 两步分解，且每步 raw integer code 严格为 `0～15`。
6. Prefix temporal K/V 注入：每一步 `K/T`、`V/T`，时间和保持一份 Prefix。
7. temporal RMSNorm、Softmax、SiLU、QK/PV seq_matmul、MLP symmetric Hadamard、final norm 的现有公式。
8. common Clip 的 cumulative-then-difference temporal 语义。
9. ANN training 和 post-finetuning calibration 的当前协议。
10. `calibration.num_samples: 128`。
11. Qwen3-1.7B TL;DR 快速评估的 `evaluation.tldr_test_samples: 128`。
12. 当前 `prefix_enabled: true` 的实验设置。
13. 训练样本数、学习率、epoch、gradient accumulation 等实验超参数。

如果实现过程中发现需要修改上述任一项才能继续，必须停止并向用户说明，不得自行扩大范围。

---

# 1. 修正后的统一位置语义

## 1.1 问题根因

当前 TL;DR evaluation 使用：

```python
tokenizer.padding_side = "left"
```

并将不同长度 prompt 放入同一个 batch。当前 `greedy_generate()` 只传递：

```python
input_ids
attention_mask
```

没有构造逐样本 `position_ids`。

当 Prefix KV 长度为 `P` 时，Prefix cache 的位置固定为：

```text
0, 1, ..., P-1
```

若让 Hugging Face 模型从共享的一维 `cache_position` 自动生成当前 token 位置，则一个左侧 padding 数量为 `pad_i` 的样本，其第一个真实 token 会被错误放到：

```text
P + pad_i
```

而不是：

```text
P
```

因此 Prefix 与当前真实 token 之间的 RoPE 相对位置会随 batch 组成改变。这个错误同时影响：

```text
ANN evaluation
Phase SNN evaluation
GIF SNN evaluation
MTN SNN evaluation
TL;DR greedy generation
lm-eval scoring/generation 中的变长 batch
```

## 1.2 唯一允许的位置约定

从本次修改开始，调用模型前的“当前序列 position IDs”统一定义为：

```text
从当前序列 attention_mask 逐样本计算；
每个样本的第一个真实 token 为 0；
后续真实 token 依次为 1、2、...；
padding token 的 position ID 取 0，因其被 mask，不参与有效注意力。
```

示例：

```text
attention_mask:
[[0, 0, 1, 1],
 [1, 1, 1, 1]]

current position_ids:
[[0, 0, 0, 1],
 [0, 1, 2, 3]]
```

Prefix wrapper 再统一加一次 `P`：

```text
model-visible current position_ids:
[[P, P, P, P+1],
 [P, P+1, P+2, P+3]]
```

必须满足：

```text
Prefix position offset 只加一次；
不除以 T；
不随 timestep 累加；
不把 padding 数量计入真实 token 的逻辑位置；
ANN 与 SNN 使用同一份逻辑 position IDs。
```

## 1.3 新增位置 helper

建议在 `snn2/evaluation.py` 新增一个小型纯函数，例如：

```python
def position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            "Position IDs require a 2-D [B, L] attention mask"
        )
    visible = attention_mask.to(dtype=torch.long)
    position_ids = visible.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(visible == 0, 0)
    return position_ids
```

行为要求：

1. 输入必须是 `[B,L]`。
2. 输出 dtype 必须为 `torch.long`。
3. 输出 device 与 mask 相同。
4. 左填充、右填充都应得到真实 token 从 0 开始的连续位置。
5. 不在这个 helper 中加入 Prefix 长度；Prefix offset 仍由 `install_prefix_kv_forward()` 负责。

这样可以保持职责清晰：

```text
evaluation：生成 current-token logical positions
prefix wrapper：加入固定 Prefix offset
temporal forward：只复制时间帧，不修改数值
```

---

# 2. 修改 `snn2/evaluation.py`

## 2.1 `greedy_generate()` 每一步生成 position IDs

在每次模型 forward 前，根据当前 `mask` 重新生成：

```python
position_ids = position_ids_from_attention_mask(mask)
```

ANN 路径必须传递：

```python
logits = model(
    input_ids=generated,
    attention_mask=mask,
    position_ids=position_ids,
    use_cache=False,
).logits
```

SNN 路径必须传递：

```python
logits = temporal_forward(
    model,
    controller,
    generated,
    mask,
    position_ids=position_ids,
)
```

注意：

1. 当前位置必须根据扩展后的 generation mask 每一步重算。
2. 不得把上一轮 position IDs 简单拼接而不处理 padding。
3. `greedy_generate()` 持有的 mask 不含虚拟 Prefix；Prefix wrapper 会在模型调用边界扩展 mask 并加位置偏移。
4. Prefix disabled 时同一个 helper 仍然可用。

## 2.2 `EvaluationModelProxy.forward()` 不得丢弃 position IDs

当前 proxy 接收 `**kwargs`，但模型调用没有传递位置相关参数。改为：

1. 若 `kwargs` 中已有 `position_ids`，验证其 batch/length 与 `input_ids` 兼容并使用它。
2. 若没有，则从当前 `attention_mask` 生成。
3. ANN 与 SNN 两条路径均传递同一逻辑 position IDs。
4. 不要把 lm-eval 的无关 kwargs 全量盲传给底层模型；只提取明确支持的模型参数。

推荐结构：

```python
position_ids = kwargs.get("position_ids")
if position_ids is None:
    position_ids = position_ids_from_attention_mask(prefixed_mask)

if self.controller.mode.startswith("deploy_"):
    logits = temporal_forward(
        self.model,
        self.controller,
        prefixed_ids,
        prefixed_mask,
        position_ids=position_ids,
    )
else:
    logits = self.model(
        input_ids=prefixed_ids,
        attention_mask=prefixed_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
```

这里变量名 `prefixed_mask` 是现有兼容命名；它在 proxy 层实际仍是“当前 token mask”，真正 Prefix mask 由模型 wrapper 注入。

## 2.3 `EvaluationModelProxy.generate()`

`generate()` 最终走 `greedy_generate()`，因此只要 `greedy_generate()` 每一步正确重建位置即可。

如果保留调用者传入的初始 `position_ids`，必须定义清晰更新协议；本次最简单且最稳妥的实现是让 `greedy_generate()` 始终从当前 mask 重建，不使用外部旧位置缓存。

---

# 3. 修改 `snn2/model_integration.py`

## 3.1 temporal batch 扩展 position IDs

当前：

```python
repeated_ids = input_ids.repeat(steps, 1)
repeated_mask = attention_mask.repeat(steps, 1)
```

这是正确的 time-major `[T*B,L]` 布局，必须保留。

但 `position_ids=[B,L]` 也必须以同样顺序扩展为：

```python
repeated_position_ids = position_ids.repeat(steps, 1)
```

不得使用：

```python
repeat_interleave(steps, dim=0)
```

因为那会得到 sample-major，而当前所有 temporal helper 都要求 time-major。

## 3.2 推荐新增 batch-kwarg helper

为了避免未来再次漏掉，可新增内部 helper，例如：

```python
def _repeat_temporal_batch_tensor(
    value: torch.Tensor,
    *,
    steps: int,
    batch: int,
    name: str,
) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError(f"{name} cannot be scalar")
    if value.shape[0] == steps * batch:
        return value
    if value.shape[0] == batch:
        return value.repeat(steps, *([1] * (value.ndim - 1)))
    if value.shape[0] == 1:
        # 允许模型原生 broadcast；也可明确 expand 后再 reshape。
        return value
    raise ValueError(
        f"{name} leading dimension is incompatible with T={steps}, B={batch}"
    )
```

本次至少对 `position_ids` 使用该逻辑。

## 3.3 `cache_position` 不按 batch/time 重复

Hugging Face 的 `cache_position` 通常是一维 `[L]`，描述序列位置而非 batch 位置。因此：

```text
position_ids [B,L]：按 T 重复
cache_position [L]：保持一份
```

若调用者传入 `cache_position`，Prefix wrapper 仍只加一次 `P`。不得对它 `.repeat(T, ...)`。

## 3.4 修改后的 `temporal_forward()` 合同

`temporal_forward()` 必须满足：

```text
input_ids:      [B,L]
attention_mask: [B,L]
position_ids:   可选 [B,L] 或已展开 [T*B,L]
cache_position: 可选 [L]

模型实际输入：
input_ids:      [T*B,L]，time-major
attention_mask: [T*B,L]，每个时间帧相同
position_ids:   [T*B,L]，每个时间帧相同
cache_position: [L]，保持固定
```

对任何不兼容 shape 必须 fail closed，不得依赖错误 broadcast。

---

# 4. 修改 `snn2/prefix_cache.py`

## 4.1 保留现有 Prefix/T 实现

以下现有行为必须保留：

```text
ANN：每个逻辑样本使用一份完整 Prefix K/V
SNN：每个 temporal frame 使用 Prefix K/T、V/T
布局：time-major [T*B,...]
Prefix attention mask：完整可见，不除以 T
```

## 4.2 明确 position contract

`install_prefix_kv_forward()` 应继续负责：

```python
kwargs["position_ids"] = kwargs["position_ids"] + cached_prefix_length
kwargs["cache_position"] = kwargs["cache_position"] + cached_prefix_length
```

但应补充清晰注释和检查：

1. 输入 `position_ids` 是当前 token 的 0-based logical positions。
2. wrapper 只加入一次 Prefix offset。
3. temporal frame 之间 position IDs 必须完全相同。
4. 不允许除以 T 或累计 T 次。

不建议同时在 `greedy_generate()` 和 Prefix wrapper 都加 `P`；只能由 Prefix wrapper 加。

## 4.3 不建议仅在 Prefix wrapper 中临时修补左填充

不要只在 Prefix wrapper 内根据“已经扩展后的 mask”猜 position IDs，因为：

1. Prefix disabled 的 batch 也应该具有稳定位置语义。
2. proxy/temporal 层仍可能丢失位置张量。
3. 职责容易再次漂移。

正确分层仍是：

```text
evaluation 生成 current positions
temporal_forward 复制 batch positions
prefix wrapper 加 P 一次
```

---

# 5. calibration 全层完整性与新工件版本

## 5.1 为什么必须提升工件格式版本

本次 calibration manifest 将新增强制字段：

```text
expected_num_hidden_layers
expected_layer_names 或等价可验证信息
```

conversion metadata 的校验合同也会增强。旧 manifest/descriptor 不包含这些保证，因此不得继续以相同 format version 被接受。

只提升：

```python
CALIBRATION_MANIFEST_FORMAT_VERSION = 3
CONVERSION_METADATA_FORMAT_VERSION = 3
```

保持：

```python
TEMPORAL_IMPLEMENTATION_VERSION = 2
SITE_STATE_FORMAT_VERSION = 2
```

原因：

```text
temporal 算术没有改变；
Phase/GIF/MTN/Clip state tensor schema 没有改变；
改变的是 manifest 和 conversion descriptor 的完整性合同。
```

旧 v2 manifest/descriptor 必须给出明确的重新生成提示。

## 5.2 production calibration 必须写入预期层数

在 `collect_site_statistics()` 中，从实际加载的模型读取：

```python
expected_num_hidden_layers = int(model.config.num_hidden_layers)
```

要求：

1. 必须是正整数。
2. 写入 `statistics_manifest.json`。
3. 写入 `calibration_state_manifest.json`。
4. 不允许从已经收集到的目录数量反推并冒充模型预期层数。

即：

```text
expected 层数来源 = 模型 config
actual 层数来源 = calibration 目录
二者必须独立，然后比较
```

## 5.3 修改 `materialize_calibration_states()`

建议增加强制关键字参数：

```python
def materialize_calibration_states(
    site_root,
    cfg,
    metadata=None,
    *,
    include_clip: bool,
    expected_num_hidden_layers: int,
):
```

或者要求 `metadata` 中必须含有该字段。无论采用哪种 API，都不得用实际目录数作为默认值让缺字段工件通过。

完成 state 写入前或 manifest 落盘前，验证实际层目录严格等于：

```python
{
    f"layer_{index:03d}"
    for index in range(expected_num_hidden_layers)
}
```

必须同时拒绝：

```text
缺少中间层，例如 layer_000、layer_002
只包含模型部分层
多出 layer_N
错误命名
某层少 site
某层多 site
```

## 5.4 集中扩展 `validate_site_topology()`

建议在 `snn2/sites.py` 中让拓扑验证接收可选的预期层数：

```python
def validate_site_topology(
    root: str | Path,
    *,
    expected_num_hidden_layers: int | None = None,
) -> dict[str, set[str]]:
```

当提供预期层数时：

```python
expected_layers = {
    f"layer_{index:03d}"
    for index in range(expected_num_hidden_layers)
}
actual_layers = {path.name for path in layers}
```

不一致时错误必须报告：

```text
missing_layers
unexpected_layers
expected_num_hidden_layers
actual_num_hidden_layers
```

不要在 `conversion.py`、`state_validation.py`、`verify_artifacts.py` 分别复制三套层名比较逻辑。

## 5.5 修改 `validate_site_state_bundle()`

该函数必须：

1. 从 v3 calibration manifest 读取 `expected_num_hidden_layers`。
2. 验证它是正整数而非 bool。
3. 用该值调用集中式 topology validator。
4. 若调用方另外提供 ANN config 的 expected layers，则两者也必须一致。
5. 再遍历所有 layer/site 加载 Phase/GIF/MTN/Clip state。
6. 保留现有跨 site timestep 一致性检查。

推荐签名：

```python
def validate_site_state_bundle(
    site_root,
    manifest=None,
    *,
    require_clip: bool,
    expected_num_hidden_layers: int | None = None,
):
```

返回结果增加：

```python
{
    "expected_num_hidden_layers": expected_layers,
    "layers": actual_layers,
    ...
}
```

## 5.6 conversion 与 ANN config 交叉验证

在 `snn2/conversion.py` 中读取：

```text
<ann_checkpoint>/config.json
```

从中取得：

```python
num_hidden_layers
```

然后把它作为 expected layers 传给 calibration bundle validator。

因此 conversion descriptor 只有在以下三者一致时才能创建或通过评估验证：

```text
ANN config.num_hidden_layers
calibration manifest.expected_num_hidden_layers
实际 layer_*/site_* 目录
```

---

# 6. rotation state 与 conversion descriptor 强绑定

## 6.1 修改 `validate_conversion_metadata()`

当前 descriptor 已保存：

```text
rotation_enabled
rotation_state_sha256
```

但 validator 没有比较 hash。必须增加：

```python
rotation_path = layout.rotation_dir / "rotation_state.pt"
```

当 `cfg["rotation"]["enabled"] is True`：

1. `rotation_state.pt` 必须存在。
2. descriptor 的 `rotation_state_sha256` 必须存在。
3. 必须等于当前文件的 `sha256_file(rotation_path)`。

当 rotation disabled：

1. descriptor 的 `rotation_state_sha256` 必须严格为 `None`。
2. 不要因共享 artifact 目录里偶然存在旧 rotation 文件而启用它。

将该字段加入统一 `expected` mapping，不能只写一个孤立的 if。

## 6.2 `create_conversion()`

创建 descriptor 前显式验证：

```text
rotation enabled -> rotation_state.pt 存在
rotation disabled -> metadata hash 为 null
```

继续保存实际 hash。

同时把新字段写入 v3 conversion metadata：

```text
expected_num_hidden_layers
```

其值必须来自 ANN config，并且已与 calibration manifest、实际目录交叉验证。

## 6.3 `verify_artifacts.py` 不得维护弱化的重复 validator

当前 `verify_artifacts.py` 手工检查 conversion metadata 的一部分字段，容易遗漏：

```text
rotation hash
prefix token IDs/hash
ANN config hash
未来的新字段
```

修改为导入并调用生产 validator：

```python
from snn2.conversion import (
    validate_calibration,
    validate_conversion_metadata,
)
```

对每个 neuron：

```python
metadata = validate_conversion_metadata(cfg, layout, neuron)
```

之后 `verify_artifacts.py` 只保留它特有的检查，例如：

```text
SNN metrics 与 descriptor 的 neuron/T/policy 一致
结果文件存在
TL;DR selection 一致
```

不要继续保留一套比生产 validator 更弱的 conversion metadata 判断。

## 6.4 本次不强制哈希大模型全部权重

本次必须检查 rotation、Prefix、ANN config、calibration manifest 的现有 hash 合同，但不擅自引入对所有 `.safetensors` 大文件逐字节哈希的高开销协议。

如果未来需要把 descriptor 与 ANN 权重完全绑定，应另行设计 checkpoint index/shard manifest；不要在本任务里临时实现一个不完整版本。

---

# 7. 修正 generated config 测试

## 7.1 保持 `configs/generated/` 不跟踪

当前 `.gitignore` 忽略：

```text
configs/generated/
```

该设计可以保留。不要为了让测试通过重新提交 12 个生成文件。

## 7.2 将 materializer 提取为可测试函数

修改 `scripts/materialize_configs.py`，提取纯函数，例如：

```python
def materialize_configs(
    matrix_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    ...
    if count != 12:
        raise RuntimeError(...)
    return generated_paths
```

`main()` 只负责 argparse 和打印路径。

要求：

1. 生成逻辑只有一份。
2. 测试和 CLI 使用同一函数。
3. 不依赖 repo 内预先存在 `configs/generated/*.yaml`。
4. 输出目录由调用方提供。

## 7.3 重写 `tests/test_generated_configs.py`

使用 `tmp_path`：

```python
@pytest.fixture()
def generated_configs(tmp_path):
    return materialize_configs(
        ROOT / "configs" / "experiment_matrix.yaml",
        tmp_path / "generated",
    )
```

然后所有测试使用该 fixture：

```text
恰好 12 个配置
全部 validate_config 通过
全部带 temporal v2 policy
GIF qmax30 / temporal_steps2 / per_step_qmax15
Qwen3-1.7B tldr_test_samples 仍为 128
篡改 legacy policy 后 validate_config 拒绝
```

干净 clone 中直接 pytest 必须能通过，无需先手工运行 materializer。

---

# 8. 修正跨平台路径测试

以下只修改测试断言，不修改 production ArtifactLayout：

```text
tests/test_post_finetuning_protocol.py
tests/test_rotated_pre_finetuning_protocol.py
```

不要使用：

```python
str(path).endswith("a/b/c")
"a/b/c" in str(path)
```

因为 Windows 使用反斜杠。

改用：

```python
path.parts[-N:] == (...)
```

或：

```python
expected = Path("a") / "b" / "c"
path == expected
```

这样 Linux/Windows 都可运行。

不要改变现有目录名，包括代码中当前使用的：

```text
prefix_enabled_ture
```

虽然 `ture` 是既有拼写，但它已经属于 artifact 路径协议；本任务不得顺手重命名，否则会造成大量工件路径漂移。

---

# 9. 必须新增或增强的测试

## 9.1 位置 helper 单测

新增到 `tests/test_temporal_prefix.py` 或新建 `tests/test_evaluation_position_ids.py`。

至少覆盖：

```text
左填充 B=2
右填充 B=2
无 padding
输出 dtype=torch.long
输出 device 不变
非法 1-D/3-D mask 拒绝
```

核心断言示例：

```python
mask = torch.tensor([
    [0, 0, 1, 1],
    [1, 1, 1, 1],
])
expected = torch.tensor([
    [0, 0, 0, 1],
    [0, 1, 2, 3],
])
```

## 9.2 temporal position time-major 测试

验证：

```text
T=2/4
B=1/3
position_ids [B,L]
展开后 [T*B,L]
每个 temporal frame 与原 position_ids 完全相同
```

使用唯一编号证明顺序为：

```text
t0:b0, t0:b1, ..., t1:b0, t1:b1, ...
```

## 9.3 Prefix offset 一次性测试

现有 test 只检查传入 position IDs 后加 Prefix 长度。扩展为同时检查：

```text
ANN：加 P 一次
SNN T=2/4：每个 frame 都加相同 P
不出现 P/T
不出现 t*P
cache_position 仍是一维并只加 P 一次
```

## 9.4 最关键的 batch invariance 回归测试

这是本次必须新增、且旧代码必须失败的测试。

构造同一个短样本：

```text
alone: [5, 6]
```

以及变长左填充 batch：

```text
batch row 0: [pad, pad, 5, 6], mask=[0,0,1,1]
batch row 1: [10, 11, 12, 13], mask=[1,1,1,1]
```

安装一个固定非空 Prefix KV，比较：

```text
短样本单独运行最后真实 token logits
短样本在左填充 batch 中运行最后真实 token logits
```

二者必须在合理数值容差内相等。

测试至少覆盖：

```text
ANN identity path
temporal bypass path，T=2/4、B=2
```

推荐使用随机初始化的小型 Hugging Face config，不下载权重：

```text
Qwen3ForCausalLM + tiny Qwen3Config
LlamaForCausalLM + tiny LlamaConfig
```

至少 Qwen3 必须覆盖，因为它含 q_norm/k_norm；Llama 可作为参数化第二模型。

测试依赖项目已锁定的：

```text
transformers==4.53.2
```

不得联网下载模型。

## 9.5 conversion rotation hash 测试

扩展 `tests/test_conversion_metadata.py`：

1. rotation enabled 时准备一个临时 `rotation_state.pt`。
2. 创建正确 hash 的 metadata，validator 应通过。
3. 修改 rotation 文件内容但不更新 descriptor，validator 必须失败。
4. 只篡改 descriptor hash，必须失败。
5. rotation disabled 但 descriptor hash 非 null，必须失败。
6. `verify_artifacts` 应复用同一个 validator 行为。

旧代码必须至少在第 3/4 项失败，证明测试确实保护新行为。

## 9.6 calibration 全层完整性测试

扩展：

```text
tests/test_calibration_topology.py
tests/test_controller_state_loading.py
tests/test_conversion_metadata.py
```

至少覆盖：

```text
expected 2 层，实际 layer_000 + layer_001 -> 通过
expected 2 层，只有 layer_000 -> 拒绝
expected 2 层，layer_000 + layer_002 -> 拒绝
manifest expected=2，ANN config num_hidden_layers=3 -> 拒绝
某层 10 个 site 不完整 -> 拒绝
所有层 neuron T 一致 -> 通过
某一 site neuron T 不一致 -> 继续拒绝
```

## 9.7 工件版本测试

必须验证：

```text
calibration manifest v2 被新代码拒绝
conversion metadata v2 被新代码拒绝
site state v2 继续有效
temporal implementation version 仍为 2
```

错误提示应明确区分：

```text
manifest/descriptor schema 已升级
temporal arithmetic 没有升级
需要重新 materialize calibration/conversion artifacts
```

## 9.8 干净 clone 测试

测试前确保：

```bash
rm -rf configs/generated
```

执行删除前必须确认目标精确为仓库内的 `configs/generated`；不得对含糊路径或仓库根递归删除。也可在全新 clone 中测试以避免删除。

然后直接：

```bash
pytest -q
```

`tests/test_generated_configs.py` 不得因目录不存在失败。

---

# 10. 推荐修改文件清单

服务器端 Codex至少检查并按需修改以下文件：

| 文件 | 必须完成的修改 |
|---|---|
| `snn2/evaluation.py` | position helper；greedy ANN/SNN 传 position IDs；proxy forward 不丢失位置 |
| `snn2/model_integration.py` | temporal time-major 重复 batch position IDs；cache_position 保持固定 |
| `snn2/prefix_cache.py` | 保留 Prefix/T；明确/验证 offset-once contract |
| `snn2/temporal_ops.py` | calibration/conversion metadata format version 升到 3；其它版本不变 |
| `snn2/sites.py` | 集中验证完整且连续的 layer 集合 |
| `snn2/calibration.py` | 从实际 model config 写 expected_num_hidden_layers；materialize 前验证完整层集合 |
| `snn2/state_validation.py` | manifest expected layers、实际层目录、可选 ANN config 层数交叉验证 |
| `snn2/conversion.py` | 读取 ANN config 层数；验证 rotation hash；写 v3 descriptor |
| `scripts/verify_artifacts.py` | 调用统一 conversion validator，不再维护弱化副本 |
| `scripts/materialize_configs.py` | 提取可测试的 materialize 函数 |
| `tests/test_temporal_prefix.py` | position/Prefix offset/time-major 测试 |
| `tests/test_temporal_model_integration.py` | tiny Qwen/Llama 左填充 batch invariance |
| `tests/test_conversion_metadata.py` | rotation hash、层数、v2 拒绝测试 |
| `tests/test_calibration_topology.py` | 缺层、跳层、多层完整性测试 |
| `tests/test_generated_configs.py` | tmp_path 动态生成 12 配置 |
| `tests/test_post_finetuning_protocol.py` | 跨平台 Path 断言 |
| `tests/test_rotated_pre_finetuning_protocol.py` | 跨平台 Path 断言 |
| `实验执行总结.md` | 记录 position 语义、新工件版本和重跑要求 |
| `代码结构总结.md` | 更新 evaluation→temporal→Prefix 的位置数据流 |

若实际代码结构变化，可调整文件归属，但所有行为和测试要求必须实现。

---

# 11. 实现顺序

严格建议按以下顺序进行，便于定位回归：

```text
1. 新增 position_ids_from_attention_mask 单测和 helper
2. 修正 greedy_generate / EvaluationModelProxy
3. 修正 temporal_forward 的 time-major position 扩展
4. 扩展 Prefix position tests
5. 增加 tiny Qwen/Llama 左填充 batch invariance 测试
6. 提升 calibration/conversion metadata format 到 v3
7. 增加 expected_num_hidden_layers 与连续层验证
8. 增加 rotation hash validator
9. 让 verify_artifacts 复用生产 validator
10. 重构 materialize_configs 测试入口
11. 修正跨平台路径断言
12. 更新两份 Markdown 文档
13. 运行定向测试
14. 运行干净 clone 全量 pytest
15. 做服务器已有本地 Qwen3 checkpoint smoke test
```

不要先大面积改代码再一次性补测试。

---

# 12. 测试命令与验收标准

## 12.1 静态检查

```bash
python -m compileall -q snn2 scripts tests
```

## 12.2 定向测试

```bash
pytest -q \
  tests/test_temporal_ops.py \
  tests/test_temporal_prefix.py \
  tests/test_temporal_model_integration.py \
  tests/test_neurons.py \
  tests/test_controller_state_loading.py \
  tests/test_calibration_topology.py \
  tests/test_conversion_metadata.py \
  tests/test_generated_configs.py \
  tests/test_post_finetuning_protocol.py \
  tests/test_rotated_pre_finetuning_protocol.py
```

## 12.3 全量测试

```bash
pytest -q
```

要求：

```text
0 failed
不依赖 repo 中预先存在 configs/generated
不联网下载模型
Linux 路径测试通过
Windows 路径断言不再依赖分隔符
```

## 12.4 tiny model 验收

至少报告：

```text
Qwen3 ANN Prefix 左填充 batch invariance max error
Qwen3 temporal bypass Prefix 左填充 batch invariance max error
Llama 对应结果（若测试已参数化）
```

合理容差可参考：

```text
float32: rtol/atol 约 1e-5
bf16: 根据算子累计误差放宽，但不得只检查 top1
```

## 12.5 服务器本地 checkpoint smoke test

如果服务器已有 Qwen3-1.7B checkpoint，选择两个不同长度短 prompt，执行：

```text
短 prompt 单独运行
短 prompt 与长 prompt 左填充组成 B=2 运行
```

比较短 prompt 最后有效 token logits。

分别执行：

```text
ANN + Prefix
至少一种 temporal bypass/真实 SNN + Prefix
```

不得为了 smoke test 下载新大模型；如果服务器没有本地 checkpoint，在交付中标记未执行原因。

---

# 13. 文档更新要求

## 13.1 `实验执行总结.md`

保持现有结构和书写风格，补充以下内容：

1. 左填充 batch 的 `position_ids` 从 attention mask 逐样本生成。
2. 当前真实 token 从 0 开始，Prefix wrapper 再加 P 一次。
3. SNN position IDs 按 time-major 重复，不除以 T、不累计。
4. cache_position 保持一份并只加 P 一次。
5. calibration manifest format v3。
6. conversion metadata format v3。
7. site state 与 temporal implementation 仍为 v2。
8. conversion 必须验证 rotation hash。
9. calibration 必须覆盖模型完整、连续的所有层。
10. `configs/generated` 仍由第一步 materialize，不提交到 Git。

继续保留并明确：

```text
calibration samples = 128
Qwen3-1.7B TL;DR quick evaluation samples = 128
prefix_enabled = true
GIF high qmax = 30
GIF two steps, each code 0..15
```

## 13.2 `代码结构总结.md`

在 evaluation 数据流中写明：

```text
attention_mask
  -> per-sample current position_ids
  -> temporal time-major repeat
  -> Prefix wrapper adds fixed P once
  -> model/RoPE
```

并写明 conversion validation 的来源绑定：

```text
ANN config num layers
calibration manifest expected layers
actual calibration layer/site directories
rotation_state hash
Prefix state/KV hash
conversion descriptor
```

---

# 14. 旧工件和重跑范围

## 14.1 旧工件处理

由于 calibration manifest 和 conversion metadata 升到 v3：

```text
旧 v2 calibration manifest 不再用于新 conversion
旧 v2 conversion descriptor 不再用于新 SNN evaluation
```

建议将旧目录移动到带 commit/date 的备份目录，不要直接混用。

移动前必须通过 `ArtifactLayout` 或明确绝对路径确认目标；不得对仓库根、`$HOME` 或含糊变量递归移动/删除。

## 14.2 必须重跑

position IDs 修正会改变 Prefix=true 的批量 evaluation，因此必须重跑：

```text
四种 ANN mode 的 ANN evaluation
四种 ANN mode × Phase/GIF/MTN 的 SNN evaluation
全部 conversion descriptor
verify_artifacts
```

由于 manifest schema 升级，还必须重新生成相应 calibration manifest。最稳妥的完整流程是按更新后的 `实验执行总结.md` 从 materialize configs 开始执行 post-finetuning calibration/conversion 阶段。

## 14.3 通常不需要因本次位置修正重训 ANN

本次 position bug 位于左填充批量 evaluation，训练和 calibration 当前通常使用 batch size 1/right padding。因此仅因 position 修正本身，不要求重新训练 ANN checkpoint。

但如果服务器尚未按 temporal v2 方案重训 phase-aware/gif-aware，仍应执行原计划中的 aware 重训；本文不改变原重跑决定。

---

# 15. 最终验收清单

服务器端 Codex 交付前逐项确认：

## 15.1 Prefix/position

```text
[ ] 左填充真实 token position 从 0 开始
[ ] Prefix offset 只加一次
[ ] SNN position_ids 使用 time-major repeat
[ ] attention mask 固定，不除以 T
[ ] cache_position 不按 T 重复
[ ] 同一样本 alone/batched logits 一致
```

## 15.2 temporal/GIF 未漂移

```text
[ ] Prefix K/T、V/T 保持不变
[ ] temporal RMSNorm/Softmax/SiLU/QK/PV/Hadamard/final norm 保持不变
[ ] GIF qmax=30
[ ] GIF 两步 code 均在 0..15
[ ] common Clip temporal 语义保持不变
```

## 15.3 工件

```text
[ ] calibration manifest v3
[ ] conversion metadata v3
[ ] site state v2
[ ] temporal implementation v2
[ ] rotation hash 在 evaluation 前验证
[ ] ANN config/manifest/实际层数一致
[ ] 层编号连续且覆盖 0..N-1
[ ] verify_artifacts 复用 production validator
```

## 15.4 测试

```text
[ ] 干净 clone 无 generated configs 时测试可运行
[ ] materializer 测试生成恰好 12 个配置
[ ] 128 快速评估配置未改变
[ ] 定向测试 0 failed
[ ] 全量 pytest 0 failed
[ ] tiny Qwen3 Prefix 左填充回归通过
[ ] 未下载任何大模型
```

## 15.5 交付说明

最终回复必须报告：

1. 修改的文件列表。
2. position IDs 的最终数据流。
3. Prefix offset 如何保证只应用一次。
4. calibration/conversion 版本变化。
5. rotation hash 与完整层验证如何实现。
6. 实际测试命令和结果。
7. tiny Qwen/Llama batch invariance 最大误差。
8. 是否执行真实本地 checkpoint smoke test。
9. 哪些实验/工件需要重新运行。
10. 明确声明未修改 128 samples、GIF qmax30、Prefix/T 和 neuron 编码方案。

---

# 16. 完成定义

只有同时满足以下条件，本任务才算完成：

```text
相同 Prefix=true 样本的结果不再依赖左填充 batch 组成；
ANN 与 temporal SNN 使用一致的逐样本逻辑位置；
Prefix position offset 只应用一次；
conversion 无法接受不匹配的 rotation state；
calibration 无法接受缺层、跳层或与 ANN config 层数不符；
干净 clone 不需要预生成配置即可运行测试；
全部测试通过；
实验文档准确反映新行为和重跑范围。
```

不得以“公式单测通过”代替 batch invariance、工件篡改拒绝和完整层验证。
