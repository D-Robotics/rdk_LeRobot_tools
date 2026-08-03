#!/usr/bin/env python3

import math
from dataclasses import asdict, dataclass

import torch

from lerobot.optim.optimizers import OptimizerConfig, OptimizerParams


class MemoryEfficientAdafactor(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-5,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        max_chunk_elements=8_388_608,
    ):
        if beta1 is not None:
            raise ValueError("MemoryEfficientAdafactor requires beta1=None")
        if scale_parameter:
            raise ValueError("MemoryEfficientAdafactor requires scale_parameter=False")
        if relative_step:
            raise ValueError("MemoryEfficientAdafactor requires relative_step=False")
        if warmup_init:
            raise ValueError("MemoryEfficientAdafactor requires warmup_init=False")
        if lr is None or lr <= 0:
            raise ValueError("MemoryEfficientAdafactor requires a positive explicit lr")
        if max_chunk_elements <= 0:
            raise ValueError("max_chunk_elements must be positive")
        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "max_chunk_elements": max_chunk_elements,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _slices(tensor: torch.Tensor, max_chunk_elements: int):
        if tensor.ndim == 0:
            yield None
            return
        trailing_elements = tensor[0].numel() if tensor.shape[0] else 1
        rows_per_chunk = max(1, max_chunk_elements // max(1, trailing_elements))
        for start in range(0, tensor.shape[0], rows_per_chunk):
            yield slice(start, min(start + rows_per_chunk, tensor.shape[0]))

    @staticmethod
    def _initialize_state(parameter, state, factored):
        state["step"] = torch.zeros((), dtype=torch.int64)
        state["RMS"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(
                parameter.shape[:-1], dtype=torch.float32, device=parameter.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                parameter.shape[:-2] + parameter.shape[-1:], dtype=torch.float32, device=parameter.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(
                parameter.shape, dtype=torch.float32, device=parameter.device
            )

    @staticmethod
    def _update_factored_state(grad, state, beta2t, epsilon, max_chunk_elements):
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]
        update_alpha = 1.0 - beta2t

        if grad.ndim == 2:
            col_mean = torch.zeros_like(exp_avg_sq_col)
            for chunk_slice in MemoryEfficientAdafactor._slices(grad, max_chunk_elements):
                squared = grad[chunk_slice].to(dtype=torch.float32, copy=True)
                squared.square_().add_(epsilon)
                exp_avg_sq_row[chunk_slice].mul_(beta2t).add_(
                    squared.mean(dim=-1), alpha=update_alpha
                )
                col_mean.add_(squared.sum(dim=-2))
            col_mean.div_(grad.shape[-2])
            exp_avg_sq_col.mul_(beta2t).add_(col_mean, alpha=update_alpha)
            return

        for chunk_slice in MemoryEfficientAdafactor._slices(grad, max_chunk_elements):
            squared = grad[chunk_slice].to(dtype=torch.float32, copy=True)
            squared.square_().add_(epsilon)
            exp_avg_sq_row[chunk_slice].mul_(beta2t).add_(
                squared.mean(dim=-1), alpha=update_alpha
            )
            exp_avg_sq_col[chunk_slice].mul_(beta2t).add_(
                squared.mean(dim=-2), alpha=update_alpha
            )

    @staticmethod
    def _update_unfactored_state(grad, state, beta2t, epsilon, max_chunk_elements):
        exp_avg_sq = state["exp_avg_sq"]
        update_alpha = 1.0 - beta2t
        if grad.ndim == 0:
            squared = grad.to(dtype=torch.float32, copy=True).square().add_(epsilon)
            exp_avg_sq.mul_(beta2t).add_(squared, alpha=update_alpha)
            return
        for chunk_slice in MemoryEfficientAdafactor._slices(grad, max_chunk_elements):
            squared = grad[chunk_slice].to(dtype=torch.float32, copy=True)
            squared.square_().add_(epsilon)
            exp_avg_sq[chunk_slice].mul_(beta2t).add_(squared, alpha=update_alpha)

    @staticmethod
    def _factored_update(grad_chunk, row_state, col_state, row_mean=None):
        if row_mean is None:
            row_mean = row_state.mean(dim=-1, keepdim=True)
        row_factor = (row_state / row_mean.clamp_min(1e-30)).rsqrt().unsqueeze(-1)
        col_factor = col_state.unsqueeze(-2).rsqrt()
        update = grad_chunk.to(dtype=torch.float32, copy=True)
        update.mul_(row_factor).mul_(col_factor)
        return update

    @staticmethod
    def _unfactored_update(grad_chunk, exp_avg_sq_chunk):
        update = grad_chunk.to(dtype=torch.float32, copy=True)
        update.mul_(exp_avg_sq_chunk.rsqrt())
        return update

    @staticmethod
    def _update_rms(grad, state, factored, max_chunk_elements):
        sum_squares = torch.zeros((), dtype=torch.float32, device=grad.device)
        if grad.ndim == 0:
            update = MemoryEfficientAdafactor._unfactored_update(grad, state["exp_avg_sq"])
            return update.abs()

        row_mean = None
        if factored and grad.ndim == 2:
            row_mean = state["exp_avg_sq_row"].mean().reshape(1)

        for chunk_slice in MemoryEfficientAdafactor._slices(grad, max_chunk_elements):
            if factored:
                row_state = state["exp_avg_sq_row"][chunk_slice]
                col_state = state["exp_avg_sq_col"] if grad.ndim == 2 else state["exp_avg_sq_col"][chunk_slice]
                update = MemoryEfficientAdafactor._factored_update(
                    grad[chunk_slice], row_state, col_state, row_mean
                )
            else:
                update = MemoryEfficientAdafactor._unfactored_update(
                    grad[chunk_slice], state["exp_avg_sq"][chunk_slice]
                )
            update.square_()
            sum_squares.add_(update.sum())
        return (sum_squares / grad.numel()).sqrt()

    @staticmethod
    def _apply_update(parameter, grad, state, factored, step_size, weight_decay, lr, max_chunk_elements):
        if grad.ndim == 0:
            update = MemoryEfficientAdafactor._unfactored_update(grad, state["exp_avg_sq"])
            parameter_fp32 = parameter.float() if parameter.dtype != torch.float32 else parameter
            if weight_decay:
                parameter_fp32.mul_(1.0 - weight_decay * lr)
            update.mul_(step_size)
            parameter_fp32.add_(update, alpha=-1.0)
            if parameter_fp32 is not parameter:
                parameter.copy_(parameter_fp32)
            return

        row_mean = None
        if factored and grad.ndim == 2:
            row_mean = state["exp_avg_sq_row"].mean().reshape(1)

        for chunk_slice in MemoryEfficientAdafactor._slices(grad, max_chunk_elements):
            if factored:
                row_state = state["exp_avg_sq_row"][chunk_slice]
                col_state = state["exp_avg_sq_col"] if grad.ndim == 2 else state["exp_avg_sq_col"][chunk_slice]
                update = MemoryEfficientAdafactor._factored_update(
                    grad[chunk_slice], row_state, col_state, row_mean
                )
            else:
                update = MemoryEfficientAdafactor._unfactored_update(
                    grad[chunk_slice], state["exp_avg_sq"][chunk_slice]
                )

            parameter_chunk = parameter[chunk_slice]
            parameter_fp32 = (
                parameter_chunk.float() if parameter_chunk.dtype != torch.float32 else parameter_chunk
            )
            if weight_decay:
                parameter_fp32.mul_(1.0 - weight_decay * lr)
            update.mul_(step_size)
            parameter_fp32.add_(update, alpha=-1.0)
            if parameter_fp32 is not parameter_chunk:
                parameter_chunk.copy_(parameter_fp32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MemoryEfficientAdafactor does not support sparse gradients")

                state = self.state[parameter]
                factored = grad.ndim >= 2
                if not state:
                    self._initialize_state(parameter, state, factored)

                state["step"].add_(1)
                step = int(state["step"].item())
                beta2t = 1.0 - math.pow(step, group["decay_rate"])
                max_chunk_elements = group["max_chunk_elements"]
                epsilon = group["eps"][0]

                if factored:
                    self._update_factored_state(
                        grad, state, beta2t, epsilon, max_chunk_elements
                    )
                else:
                    self._update_unfactored_state(
                        grad, state, beta2t, epsilon, max_chunk_elements
                    )

                update_rms = self._update_rms(grad, state, factored, max_chunk_elements)
                state["RMS"].copy_(update_rms)
                clip_denom = torch.clamp(
                    update_rms / group["clip_threshold"], min=1.0
                )
                step_size = group["lr"] / clip_denom
                self._apply_update(
                    parameter,
                    grad,
                    state,
                    factored,
                    step_size,
                    group["weight_decay"],
                    group["lr"],
                    max_chunk_elements,
                )
        return loss


@OptimizerConfig.register_subclass("memory_efficient_adafactor")
@dataclass
class MemoryEfficientAdafactorConfig(OptimizerConfig):
    lr: float = 1e-5
    eps: tuple[float, float] = (1e-30, 1e-3)
    clip_threshold: float = 1.0
    decay_rate: float = -0.8
    beta1: float | None = None
    weight_decay: float = 1e-2
    scale_parameter: bool = False
    relative_step: bool = False
    warmup_init: bool = False
    max_chunk_elements: int = 8_388_608
    grad_clip_norm: float = 1.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return MemoryEfficientAdafactor(params, **kwargs)
