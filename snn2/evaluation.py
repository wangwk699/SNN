from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .controller import SiteController
from .model_integration import temporal_forward


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

    # [batch_size]
    # 记录 batch 中每个样本是否已经生成 EOS。
    finished = torch.zeros(
        input_ids.shape[0],
        dtype=torch.bool,
        device=input_ids.device,
    )

    for _ in range(max_new_tokens):
        if counter is not None:
            counter["model_forward_calls"] = (
                counter.get("model_forward_calls", 0) + 1
            )

            temporal = 1
            if controller is not None and controller.mode.startswith("deploy_"):
                temporal = int(controller.temporal_steps or 1)

            counter["temporal_model_step_forwards"] = (
                counter.get("temporal_model_step_forwards", 0)
                + temporal
            )

        # ANN / 非 temporal deployment
        if controller is None or not controller.mode.startswith("deploy_"):
            logits = model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=False,
            ).logits

        # Full-temporal SNN deployment
        else:
            logits = temporal_forward(
                model,
                controller,
                generated,
                mask,
            )

        # 每个样本独立选择下一个 token
        # shape: [batch_size, 1]
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        if eos_token_id is not None:
            # 对已经结束的样本，之后始终补 EOS，
            # 避免它们继续产生新的有效文本。
            eos_tokens = torch.full_like(
                next_token,
                eos_token_id,
            )

            next_token = torch.where(
                finished.unsqueeze(-1),
                eos_tokens,
                next_token,
            )

        # 拼接新生成 token
        generated = torch.cat(
            (generated, next_token),
            dim=-1,
        )

        # 新 token 对应 attention mask = 1
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
            # 更新每个样本自己的结束状态
            finished = finished | (
                next_token.squeeze(-1) == eos_token_id
            )

            # 整个 batch 都结束时提前停止
            if finished.all():
                break

    return generated


@dataclass
class ProxyOutput:
    logits: torch.Tensor
    past_key_values: None = None


class EvaluationModelProxy(nn.Module):
    """Adapter used by lm-eval so every request sees prefix and temporal SNN execution."""

    def __init__(self, model: nn.Module, controller: SiteController, prefix_ids: list[int]):
        super().__init__()
        self.model = model
        self.controller = controller
        self.prefix_ids = list(prefix_ids)
        self.config = model.config
        self.name_or_path = getattr(model, "name_or_path", "snn2-proxy")
        self.execution_counter: dict[str, int] = {}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _prefix(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not self.prefix_ids:
            return input_ids, attention_mask, 0
        prefix = torch.tensor(self.prefix_ids, device=input_ids.device, dtype=input_ids.dtype)
        prefix = prefix.unsqueeze(0).expand(input_ids.shape[0], -1)
        prefix_mask = torch.ones_like(prefix)
        return (
            torch.cat((prefix, input_ids), dim=-1),
            torch.cat((prefix_mask, attention_mask), dim=-1),
            prefix.shape[-1],
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        prefixed_ids, prefixed_mask, length = self._prefix(input_ids, attention_mask)
        if self.controller.mode.startswith("deploy_"):
            logits = temporal_forward(
                self.model, self.controller, prefixed_ids, prefixed_mask
            )
            temporal = int(self.controller.temporal_steps or 1)
        else:
            logits = self.model(
                input_ids=prefixed_ids,
                attention_mask=prefixed_mask,
                use_cache=False,
            ).logits
            temporal = 1
        self.execution_counter["model_forward_calls"] = (
            self.execution_counter.get("model_forward_calls", 0) + 1
        )
        self.execution_counter["temporal_model_step_forwards"] = (
            self.execution_counter.get("temporal_model_step_forwards", 0)
            + temporal
        )
        if length:
            logits = logits[:, length:]
        return ProxyOutput(logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        prefixed_ids, prefixed_mask, length = self._prefix(input_ids, attention_mask)
        max_new = int(kwargs.get("max_new_tokens", kwargs.get("max_length", 256) - input_ids.shape[-1]))
        result = greedy_generate(
            self.model,
            self.controller,
            prefixed_ids,
            prefixed_mask,
            max_new_tokens=max_new,
            eos_token_id=kwargs.get("eos_token_id", self.config.eos_token_id),
            counter=self.execution_counter,
        )
        return result[:, length:] if length else result

    def tie_weights(self):
        return self.model.tie_weights()
