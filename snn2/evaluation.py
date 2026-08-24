from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .artifacts import prefix_enabled_dirname
from .controller import SiteController
from .model_integration import temporal_forward
from .prefix_cache import install_prefix_kv_forward
from .sites import SITE_COUNT
from .temporal_ops import temporal_policy_metadata


def position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample 0-based logical positions for visible tokens."""
    if attention_mask.ndim != 2:
        raise ValueError("Position IDs require a 2-D [B, L] attention mask")
    visible = attention_mask.to(dtype=torch.long)
    position_ids = visible.cumsum(dim=-1) - 1
    return position_ids.masked_fill(visible == 0, 0)



def deployment_policy_metadata(controller: SiteController | None) -> dict[str, object]:
    if controller is None or not controller.mode.startswith("deploy_"):
        return {}
    return temporal_policy_metadata()


def activation_neuron_operators_per_temporal_forward(
    *, num_hidden_layers: int, neuron: str
) -> int:
    base = int(num_hidden_layers) * SITE_COUNT
    if neuron == "phase":
        base += 1
    return base



def resolve_tldr_evaluation_layout(
    total_test_samples: int,
    configured_test_samples: int | None,
) -> dict[str, object]:
    """Resolve the canonical TL;DR result directory and split cardinality."""
    if total_test_samples <= 0:
        raise ValueError("total_test_samples must be positive")
    if configured_test_samples is None:
        selected, is_full = total_test_samples, True
    else:
        requested = int(configured_test_samples)
        if requested <= 0:
            raise ValueError("evaluation.tldr_test_samples must be a positive integer or null")
        selected = min(requested, total_test_samples)
        is_full = selected == total_test_samples
    dirname = f"test_samples_{selected}_full" if is_full else f"test_samples_{selected}"
    return {
        "selected_test_samples": selected,
        "is_full_test": is_full,
        "dirname": dirname,
    }


def rotated_pre_finetuning_prefix_dirname(enabled: bool) -> str:
    """Backward-compatible alias for the general Prefix result suffix."""
    return prefix_enabled_dirname(enabled)

def _update_execution_counter(
    counter: dict[str, int] | None,
    *,
    batch_size: int,
    temporal_steps: int,
    active_samples: int | None = None,
) -> None:
    """
    Update execution statistics.

    model_forward_calls:
        Number of actual batched model forward calls.
        This depends on batch size.

    temporal_model_step_forwards:
        Logical temporal model-step equivalents across actual
        batched model forward calls. For ANN, temporal_steps = 1.
        For full-temporal SNN deployment, each model call represents
        temporal_steps logical timestep executions.

    sample_forward_equivalents:
        Logical sample-level forward equivalents.
        This is independent of evaluation batch size.

    temporal_sample_step_forwards:
        Logical sample-level temporal forward equivalents.
        This is the quantity used for the batch-size-independent
        SNN activation-site operator metric.

    batched_sample_slots:
        Number of sample slots actually present in executed batched forwards.
        Finished sequences may still occupy slots until the whole batch ends.

    batched_temporal_sample_slots:
        batched_sample_slots multiplied by temporal_steps.
    """
    if counter is None:
        return

    batch_size = int(batch_size)
    temporal_steps = int(temporal_steps)

    if active_samples is None:
        active_samples = batch_size

    active_samples = int(active_samples)

    counter["model_forward_calls"] = (
        counter.get("model_forward_calls", 0)
        + 1
    )

    counter["temporal_model_step_forwards"] = (
        counter.get("temporal_model_step_forwards", 0)
        + temporal_steps
    )

    counter["sample_forward_equivalents"] = (
        counter.get("sample_forward_equivalents", 0)
        + active_samples
    )

    counter["temporal_sample_step_forwards"] = (
        counter.get("temporal_sample_step_forwards", 0)
        + active_samples * temporal_steps
    )

    counter["batched_sample_slots"] = (
        counter.get("batched_sample_slots", 0)
        + batch_size
    )

    counter["batched_temporal_sample_slots"] = (
        counter.get("batched_temporal_sample_slots", 0)
        + batch_size * temporal_steps
    )


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    controller: SiteController | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    counter: dict[str, int] | None = None,
) -> torch.Tensor:
    generated = input_ids
    mask = attention_mask

    batch_size = int(input_ids.shape[0])

    # [batch_size]
    # 记录 batch 中每个样本是否已经生成 EOS。
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )

    for _ in range(max_new_tokens):
        position_ids = position_ids_from_attention_mask(mask)

        # 当前这个 generation step 中，
        # 还有多少样本在逻辑上需要继续生成。
        #
        # 这样计算 sample-equivalent cost 时不会因为
        # batch 中其它样本生成得更长，而重复计算已经结束的样本。
        if eos_token_id is None:
            active_samples = batch_size
        else:
            active_samples = int(
                (~finished).sum().item()
            )

        temporal = 1
        if (
            controller is not None
            and controller.mode.startswith("deploy_")
        ):
            temporal = int(
                controller.temporal_steps or 1
            )

        _update_execution_counter(
            counter,
            batch_size=batch_size,
            temporal_steps=temporal,
            active_samples=active_samples,
        )

        # ANN / 非 temporal deployment
        if (
            controller is None
            or not controller.mode.startswith("deploy_")
        ):
            logits = model(
                input_ids=generated,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

        # Full-temporal SNN deployment
        else:
            logits = temporal_forward(
                model,
                controller,
                generated,
                mask,
                position_ids=position_ids,
            )

        # 每个样本独立选择下一个 token
        # [batch_size, 1]
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        if eos_token_id is not None:
            eos_tokens = torch.full_like(
                next_token,
                eos_token_id,
            )

            # 已经结束的样本保持 EOS，
            # 避免继续产生有效文本。
            next_token = torch.where(
                finished.unsqueeze(-1),
                eos_tokens,
                next_token,
            )

        generated = torch.cat(
            (
                generated,
                next_token,
            ),
            dim=-1,
        )

        mask = torch.cat(
            (
                mask,
                torch.ones_like(
                    next_token,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )

        if eos_token_id is not None:
            finished = finished | (
                next_token.squeeze(-1)
                == eos_token_id
            )

            # 所有样本都结束后才退出 batch generation
            if finished.all():
                break

    return generated


@dataclass
class ProxyOutput:
    logits: torch.Tensor
    past_key_values: None = None


class EvaluationModelProxy(nn.Module):
    """
    Adapter used by lm-eval so every request sees
    prefix and temporal SNN execution.
    """

    def __init__(
        self,
        model: nn.Module,
        controller: SiteController,
        prefix_key_values,
    ):
        super().__init__()

        self.model = model
        self.controller = controller
        install_prefix_kv_forward(
            self.model, prefix_key_values, controller=self.controller
        )

        self.config = model.config

        self.name_or_path = getattr(
            model,
            "name_or_path",
            "snn2-proxy",
        )

        self.execution_counter: dict[str, int] = {}

    @property
    def device(self):
        return next(
            self.model.parameters()
        ).device

    def _prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        prefixed_ids, prefixed_mask, length = (
            self._prefix(
                input_ids,
                attention_mask,
            )
        )
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = position_ids_from_attention_mask(prefixed_mask)
        elif (
            position_ids.ndim != 2
            or position_ids.shape[0] not in {1, prefixed_ids.shape[0]}
            or position_ids.shape[-1] != prefixed_ids.shape[-1]
        ):
            raise ValueError(
                "position_ids must have shape [B, L] (or [1, L]) compatible "
                "with input_ids"
            )
        elif position_ids.shape[0] == 1 and prefixed_ids.shape[0] != 1:
            position_ids = position_ids.expand(prefixed_ids.shape[0], -1)

        batch_size = int(
            prefixed_ids.shape[0]
        )

        if self.controller.mode.startswith(
            "deploy_"
        ):
            logits = temporal_forward(
                self.model,
                self.controller,
                prefixed_ids,
                prefixed_mask,
                position_ids=position_ids,
            )

            temporal = int(
                self.controller.temporal_steps
                or 1
            )

        else:
            logits = self.model(
                input_ids=prefixed_ids,
                attention_mask=prefixed_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

            temporal = 1

        # lm-eval scoring forward 没有 generation EOS 问题，
        # batch 中所有样本都是有效样本。
        _update_execution_counter(
            self.execution_counter,
            batch_size=batch_size,
            temporal_steps=temporal,
            active_samples=batch_size,
        )

        if length:
            logits = logits[:, length:]

        return ProxyOutput(
            logits=logits
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        prefixed_ids, prefixed_mask, length = (
            self._prefix(
                input_ids,
                attention_mask,
            )
        )

        max_new = int(
            kwargs.get(
                "max_new_tokens",
                kwargs.get(
                    "max_length",
                    256,
                )
                - input_ids.shape[-1],
            )
        )

        result = greedy_generate(
            self.model,
            self.controller,
            prefixed_ids,
            prefixed_mask,
            max_new_tokens=max_new,
            eos_token_id=kwargs.get(
                "eos_token_id",
                self.config.eos_token_id,
            ),
            counter=self.execution_counter,
        )

        if length:
            return result[:, length:]

        return result

    def tie_weights(self):
        return self.model.tie_weights()
