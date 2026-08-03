#!/usr/bin/env python3
from __future__ import annotations

import types

import torch
from hbdk4.compiler import leap
from leap_llm.models.pi0.blocks.attention import GemmaAttention, RotaryPosEmb
from leap_llm.models.pi0.blocks.mlp import GemmaMLP
from leap_llm.models.pi0.blocks.rmsnorm import GemmaRMSNorm
from leap_llm.models.pi0.model_gemma_expert import GemmaExpert
from leap_llm.models.pi0.model_gemma_expert import Attention as ExpertAttention
from leap_llm.models.pi0.model_siglip import SiglipAttention
from leap_llm.nn.modules import FakeQuantLinear, FakeQuantMatmul
from leap_llm.nn.modules.const_fake_quant import ConstFakeQuant
from leap_llm.nn.modules.activation import FakeQuantGELU
from leap_llm.nn.modules.layer_norm import LayerNormSplit


def _apply_rotary_pos_emb_torch(self, query_states, key_states, cos, sin):
    cos = cos.to(device=query_states.device, dtype=query_states.dtype)
    sin = sin.to(device=query_states.device, dtype=query_states.dtype)
    query_embed = torch.mul(query_states, cos)
    query_embed = torch.add(
        query_embed, torch.mul(self.rotate_half_torch(query_states), sin)
    )
    key_embed = torch.mul(key_states, cos)
    key_embed = torch.add(
        key_embed, torch.mul(self.rotate_half_torch(key_states), sin)
    )
    return query_embed, key_embed


def _gemma_mlp_tanh_forward(self, hidden_state):
    gate = self.gate_proj(hidden_state)
    gate = torch.nn.functional.gelu(gate, approximate="tanh")
    up_projection = self.up_proj(hidden_state)
    return self.down_proj(torch.mul(gate, up_projection))


def _gemma_mlp_tanh_build(self, hidden_state):
    gate = self.gate_proj(hidden_state)
    gate = leap.gelu(gate, approximate="tanh")
    up_projection = self.up_proj(hidden_state)
    return self.down_proj(leap.mul(gate, up_projection))


def _fake_quant_gelu_tanh_forward(self, hidden_state):
    output = torch.nn.functional.gelu(hidden_state, approximate="tanh")
    if self.quantized:
        output = self.out_quant(output)
    return output


def _fake_quant_gelu_tanh_build(self, hidden_state):
    output = leap.gelu(hidden_state, approximate="tanh")
    if self.quantized:
        output = self.out_quant(output)
    return output


def _dynamic_linear_w8a16_build(self, hidden_state):
    hidden_state = leap.cast_type(hidden_state, output_type=leap.float16)
    weight = self.weight.data.to(torch.float32)
    absmax = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps)
    quantized_weight = leap.const_fake_quant(
        weight,
        (-absmax).tolist(),
        absmax.tolist(),
        8,
        True,
        axis=0,
    )
    quantized_weight = leap.cast_type(quantized_weight, output_type=leap.float16)
    bias = self.bias.data.to(torch.float16) if self.bias is not None else None
    return leap.linear(hidden_state, quantized_weight, bias=bias)


def _siglip_patch_embedding_fp16_build(self, hidden_state):
    hidden_state = leap.cast_type(hidden_state, output_type=leap.float16)
    weight = self.weight.data.to(torch.float16)
    bias = self.bias.data.to(torch.float16)
    return leap.conv2d(
        input=hidden_state,
        weight=weight,
        bias=bias,
        stride=(self.patch_size, self.patch_size),
    )


def _siglip_position_embedding_fp16_build(self, position_ids):
    weight = self.weight.data.to(torch.float16)
    return leap.gather_nd(weight, position_ids, 0)


def _dynamic_linear_per_block_build(self, hidden_state):
    block_size = int(self._pi0_activation_block_size)
    int_max = 2 ** (self.w_bits - 1) - 1
    if self.has_scale:
        raise ValueError("Per-block linear does not support precomputed scales")
    weight = self.weight.data
    input_shape = tuple(hidden_state.type.shape)
    if input_shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input width {input_shape[-1]} does not match weight width "
            f"{weight.shape[-1]}"
        )
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Weight width {weight.shape[-1]} is not divisible by block size {block_size}"
        )
    block_count = weight.shape[-1] // block_size
    blocked_input = leap.reshape(
        hidden_state,
        (*input_shape[:-1], block_count, 1, block_size),
    )
    quantized_input, input_scale = leap.dynamic_quantize(
        blocked_input, blockSize=-1
    )
    weight_blocks = weight.reshape(weight.shape[0], block_count, block_size)
    weight_max = weight_blocks.abs().amax(dim=-1)
    weight_scale = (weight_max / int_max).clamp_min(torch.finfo(weight.dtype).eps)
    quantized_weight = torch.round(weight_blocks / weight_scale.unsqueeze(-1))
    quantized_weight = torch.clamp(
        quantized_weight, -int_max - 1, int_max
    ).to(torch.int8)
    quantized_weight = quantized_weight.permute(1, 0, 2).contiguous()
    weight_scale = weight_scale.permute(1, 0).unsqueeze(-1).contiguous()
    blocked_output = leap.block_quantized_matmul(
        quantized_input,
        quantized_weight,
        input_scale,
        weight_scale,
        mmaAlpha=float(getattr(self, "_pi0_mma_alpha", 1024.0)),
    )
    blocked_output = leap.reshape(
        blocked_output,
        (*input_shape[:-1], block_count, weight.shape[0]),
    )
    output = leap.reduce_sum(
        blocked_output,
        dims=[len(input_shape) - 1],
        keepDim=False,
    )
    if self.bias is not None:
        output = leap.add(output, self.bias.data)
    return output


def _dynamic_linear_split_block_build(self, hidden_state):
    block_size = int(self._pi0_activation_block_size)
    weight = self.weight.data.to(torch.float32)
    input_shape = tuple(hidden_state.type.shape)
    if input_shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input width {input_shape[-1]} does not match weight width "
            f"{weight.shape[-1]}"
        )
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Weight width {weight.shape[-1]} is not divisible by block size {block_size}"
        )

    begin = [0] * len(input_shape)
    end = list(input_shape)
    step = [1] * len(input_shape)
    partials = []
    for start in range(0, weight.shape[-1], block_size):
        stop = start + block_size
        begin[-1] = start
        end[-1] = stop
        input_block = leap.slice(hidden_state, begin, end, step)
        weight_block = weight[:, start:stop].contiguous()
        absmax = weight_block.abs().amax(dim=1).clamp_min(
            torch.finfo(torch.float32).eps
        )
        quantized_weight = leap.const_fake_quant(
            weight_block,
            (-absmax).tolist(),
            absmax.tolist(),
            8,
            True,
            axis=0,
        )
        quantized_weight = leap.cast_type(
            quantized_weight, output_type=leap.float16
        )
        partial = leap.linear(input_block, quantized_weight, bias=None)
        partials.append(leap.cast_type(partial, output_type=leap.float32))

    while len(partials) > 1:
        reduced = []
        for index in range(0, len(partials), 2):
            if index + 1 == len(partials):
                reduced.append(partials[index])
            else:
                reduced.append(leap.add(partials[index], partials[index + 1]))
        partials = reduced

    output = partials[0]
    if self.bias is not None:
        output = leap.add(output, self.bias.data.to(torch.float32))
    return leap.cast_type(output, output_type=leap.float16)


def _dynamic_linear_unrolled_block_build(self, hidden_state):
    block_size = int(self._pi0_activation_block_size)
    int_max = 2 ** (self.w_bits - 1) - 1
    if self.has_scale:
        raise ValueError("Unrolled per-block linear does not support precomputed scales")
    weight = self.weight.data.to(torch.float32)
    input_shape = tuple(hidden_state.type.shape)
    if input_shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input width {input_shape[-1]} does not match weight width "
            f"{weight.shape[-1]}"
        )
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Weight width {weight.shape[-1]} is not divisible by block size {block_size}"
        )

    block_count = weight.shape[-1] // block_size
    blocked_input = leap.reshape(
        hidden_state,
        (*input_shape[:-1], block_count, 1, block_size),
    )
    block_dimension = len(input_shape) - 1
    partials = []
    for block_index in range(block_count):
        input_block = leap.select(blocked_input, block_dimension, block_index)
        quantized_input, input_scale = leap.dynamic_quantize(
            input_block, blockSize=-1
        )
        start = block_index * block_size
        stop = start + block_size
        weight_block = weight[:, start:stop].contiguous()
        weight_max = weight_block.abs().amax(dim=-1)
        weight_scale = (weight_max / int_max).clamp_min(
            torch.finfo(torch.float32).eps
        )
        quantized_weight = torch.round(weight_block / weight_scale.unsqueeze(-1))
        quantized_weight = torch.clamp(
            quantized_weight, -int_max - 1, int_max
        ).to(torch.int8)
        partial = leap.block_quantized_matmul(
            quantized_input,
            quantized_weight,
            input_scale,
            weight_scale.unsqueeze(-1),
            mmaAlpha=1.0,
        )
        partials.append(
            leap.reshape(partial, (*input_shape[:-1], weight.shape[0]))
        )

    while len(partials) > 1:
        reduced = []
        for index in range(0, len(partials), 2):
            if index + 1 == len(partials):
                reduced.append(partials[index])
            else:
                reduced.append(leap.add(partials[index], partials[index + 1]))
        partials = reduced

    output = partials[0]
    if self.bias is not None:
        output = leap.add(output, self.bias.data.to(torch.float16))
    return output


def _dynamic_linear_transposed_block_build(self, hidden_state):
    block_size = int(self._pi0_activation_block_size)
    int_max = 2 ** (self.w_bits - 1) - 1
    if self.has_scale:
        raise ValueError("Transposed per-block linear does not support precomputed scales")
    weight = self.weight.data.to(torch.float32)
    input_shape = tuple(hidden_state.type.shape)
    if len(input_shape) != 3:
        raise ValueError(f"Expected rank-3 linear input, got {input_shape}")
    if input_shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input width {input_shape[-1]} does not match weight width "
            f"{weight.shape[-1]}"
        )
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Weight width {weight.shape[-1]} is not divisible by block size {block_size}"
        )

    block_count = weight.shape[-1] // block_size
    blocked_input = leap.reshape(
        hidden_state,
        (input_shape[0], input_shape[1], block_count, block_size),
    )
    blocked_input = leap.transpose(blocked_input, [0, 2, 1, 3])
    quantized_input, input_scale = leap.dynamic_quantize(
        blocked_input, blockSize=-1
    )
    weight_blocks = weight.reshape(weight.shape[0], block_count, block_size)
    weight_max = weight_blocks.abs().amax(dim=-1)
    weight_scale = (weight_max / int_max).clamp_min(
        torch.finfo(torch.float32).eps
    )
    quantized_weight = torch.round(weight_blocks / weight_scale.unsqueeze(-1))
    quantized_weight = torch.clamp(
        quantized_weight, -int_max - 1, int_max
    ).to(torch.int8)
    quantized_weight = quantized_weight.permute(1, 0, 2).contiguous()
    weight_scale = weight_scale.permute(1, 0).unsqueeze(-1).contiguous()
    blocked_output = leap.block_quantized_matmul(
        quantized_input,
        quantized_weight,
        input_scale,
        weight_scale,
        mmaAlpha=1024.0,
    )
    blocked_output = leap.transpose(blocked_output, [0, 2, 1, 3])
    output = leap.reduce_sum(blocked_output, dims=[2], keepDim=False)
    if self.bias is not None:
        output = leap.add(output, self.bias.data.to(torch.float16))
    return output


def _dynamic_linear_quantized_transposed_block_build(self, hidden_state):
    block_size = int(self._pi0_activation_block_size)
    int_max = 2 ** (self.w_bits - 1) - 1
    if self.has_scale:
        raise ValueError(
            "Quantized-transposed per-block linear does not support precomputed scales"
        )
    weight = self.weight.data
    input_shape = tuple(hidden_state.type.shape)
    if len(input_shape) != 3:
        raise ValueError(f"Expected rank-3 linear input, got {input_shape}")
    if input_shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input width {input_shape[-1]} does not match weight width "
            f"{weight.shape[-1]}"
        )
    if weight.shape[-1] % block_size != 0:
        raise ValueError(
            f"Weight width {weight.shape[-1]} is not divisible by block size {block_size}"
        )

    block_count = weight.shape[-1] // block_size
    blocked_input = leap.reshape(
        hidden_state,
        (input_shape[0], input_shape[1], block_count, 1, block_size),
    )
    quantized_input, input_scale = leap.dynamic_quantize(
        blocked_input, blockSize=-1
    )
    quantized_input = leap.reshape(
        quantized_input,
        (input_shape[0], input_shape[1], block_count, block_size),
    )
    quantized_input = leap.transpose(quantized_input, [0, 2, 1, 3])
    input_scale = leap.reshape(
        input_scale,
        (input_shape[0], input_shape[1], block_count, 1),
    )
    input_scale = leap.transpose(input_scale, [0, 2, 1, 3])

    weight_blocks = weight.reshape(weight.shape[0], block_count, block_size)
    weight_max = weight_blocks.abs().amax(dim=-1)
    weight_scale = (weight_max / int_max).clamp_min(torch.finfo(weight.dtype).eps)
    quantized_weight = torch.round(weight_blocks / weight_scale.unsqueeze(-1))
    quantized_weight = torch.clamp(
        quantized_weight, -int_max - 1, int_max
    ).to(torch.int8)
    quantized_weight = quantized_weight.permute(1, 0, 2).contiguous()
    weight_scale = weight_scale.permute(1, 0).unsqueeze(-1).contiguous()
    blocked_output = leap.block_quantized_matmul(
        quantized_input,
        quantized_weight,
        input_scale,
        weight_scale,
        mmaAlpha=1024.0,
    )
    blocked_output = leap.transpose(blocked_output, [0, 2, 1, 3])
    output = leap.reduce_sum(blocked_output, dims=[2], keepDim=False)
    if self.bias is not None:
        output = leap.add(output, self.bias.data)
    return output


def _static_linear_w8a16_forward(self, hidden_state):
    hidden_state = self.x_fake_quant(hidden_state)
    weight = self.weight.data
    if hidden_state.dtype != weight.dtype:
        hidden_state = hidden_state.to(weight.dtype)
    bias = self.bias.data if self.bias is not None else None
    output = torch.nn.functional.linear(hidden_state, weight, bias=bias)
    return self.out_fake_quant(output)


def _static_linear_w8a16_build(self, hidden_state):
    hidden_state = self.x_fake_quant(hidden_state)
    weight = self.weight.data.to(torch.float32)
    absmax = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps)
    quantized_weight = leap.const_fake_quant(
        weight,
        (-absmax).tolist(),
        absmax.tolist(),
        8,
        True,
        axis=0,
    )
    quantized_weight = leap.cast_type(quantized_weight, output_type=leap.float16)
    bias = self.bias.data.to(torch.float16) if self.bias is not None else None
    output = leap.linear(hidden_state, quantized_weight, bias=bias)
    return self.out_fake_quant(output)


def _enable_static_linear_w8a16(module):
    module.x_fake_quant = ConstFakeQuant(16)
    module.out_fake_quant = ConstFakeQuant(16)
    module._pi0_compile_dispatch_forward = module.forward
    module.forward = types.MethodType(_static_linear_w8a16_forward, module)
    module.build = types.MethodType(_static_linear_w8a16_build, module)
    module._pi0_static_linear_w8a16 = True


def finalize_static_linear_calibration(model) -> int:
    finalized = 0
    for module in model.modules():
        if not getattr(module, "_pi0_static_linear_w8a16", False):
            continue
        module.forward = module._pi0_compile_dispatch_forward
        finalized += 1
    return finalized


def _dynamic_linear_fp16_build(self, hidden_state):
    hidden_state = leap.cast_type(hidden_state, output_type=leap.float16)
    weight = self.weight.data.to(torch.float16)
    bias = self.bias.data.to(torch.float16) if self.bias is not None else None
    return leap.linear(hidden_state, weight, bias=bias)


def _dynamic_matmul_fp16_build(self, left, right):
    rank = len(right.type.shape)
    permutation = list(range(rank))
    permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
    return leap.matmul(left, leap.transpose(right, permutation))


def _replace_siglip_layernorm(parent, attribute: str) -> bool:
    current = getattr(parent, attribute, None)
    if current is None or isinstance(current, LayerNormSplit):
        return False
    replacement = LayerNormSplit(
        current.weight.numel(),
        eps=current.eps,
        bias=current.bias is not None,
    )
    replacement.weight.data.copy_(current.weight.data)
    if current.bias is not None:
        replacement.bias.data.copy_(current.bias.data)
    setattr(parent, attribute, replacement)
    return True


def _configure_siglip_layernorms(siglip, mode: str) -> int:
    if mode == "standard":
        return 0
    if mode != "split":
        raise ValueError(f"Unsupported SigLIP LayerNorm mode: {mode}")
    replacement_count = 0
    for layer in siglip.model.encoder.layers:
        replacement_count += _replace_siglip_layernorm(layer, "layer_norm1")
        replacement_count += _replace_siglip_layernorm(layer, "layer_norm2")
    replacement_count += _replace_siglip_layernorm(siglip.model, "post_layernorm")
    head = getattr(siglip.model, "head", None)
    if head is not None:
        replacement_count += _replace_siglip_layernorm(head, "layernorm")
    return replacement_count


def _fakequant_linear_fp32_build(self, hidden_state):
    hidden_state = leap.cast_type(hidden_state, output_type=leap.float32)
    quantized_input = self.x_fake_quant(hidden_state)
    weight = self.weight.data.to(torch.float32)
    weight_absmax = self.absmax_weight.to(torch.float32)
    quantized_weight = leap.const_fake_quant(
        weight,
        (-weight_absmax).tolist(),
        weight_absmax.tolist(),
        self.w_bits,
        True,
        axis=0,
    )
    bias = self.bias.data.to(torch.float32) if self.bias is not None else None
    output = leap.linear(quantized_input, quantized_weight, bias=bias)
    return self.out_fake_quant(output)


def _fakequant_linear_fp32_output_fp16_build(self, hidden_state):
    output = _fakequant_linear_fp32_build(self, hidden_state)
    return leap.cast_type(output, output_type=leap.float16)


def _replace_siglip_linear(
    parent, attribute: str, output_fp16: bool = False
) -> bool:
    current = getattr(parent, attribute)
    if isinstance(current, FakeQuantLinear):
        build = (
            _fakequant_linear_fp32_output_fp16_build
            if output_fp16
            else _fakequant_linear_fp32_build
        )
        current.build = types.MethodType(build, current)
        return False
    replacement = FakeQuantLinear(
        current.weight.shape[1],
        current.weight.shape[0],
        bias=current.bias is not None,
        quant_bits=16,
        w_bits=8,
    )
    replacement.weight.data.copy_(current.weight.data)
    if current.bias is not None:
        replacement.bias.data.copy_(current.bias.data)
    build = (
        _fakequant_linear_fp32_output_fp16_build
        if output_fp16
        else _fakequant_linear_fp32_build
    )
    replacement.build = types.MethodType(build, replacement)
    setattr(parent, attribute, replacement)
    return True


def _fixed_quant_transposed_matmul_build(self, left, right):
    if self.x_bits:
        left = leap.cast_type(left, output_type=leap.float32)
        left = self.x_fake_quant(left)
    if self.y_bits:
        right = leap.cast_type(right, output_type=leap.float32)
        right = self.y_fake_quant(right)
    rank = len(right.type.shape)
    permutation = list(range(rank))
    permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
    output = leap.matmul(
        left,
        leap.transpose(right, permutation),
        output_type=leap.float32,
    )
    if self.out_fake_quant is not None:
        output = self.out_fake_quant(output)
    return leap.cast_type(output, output_type=leap.float16)


def configure_paligemma_export_precision(
    paligemma,
    attention_linear_mode: str = "dynamic",
    attention_linear_last_layer_count: int | None = None,
    mlp_linear_mode: str = "dynamic",
    mlp_down_linear_mode: str | None = None,
    mlp_down_linear_layer_count: int | None = None,
    attention_matmul_mode: str = "dynamic",
    fp16_last_layers: int = 0,
) -> dict:
    linear_builds = {
        "dynamic": None,
        "block32": _dynamic_linear_per_block_build,
        "block64": _dynamic_linear_per_block_build,
        "block128": _dynamic_linear_per_block_build,
        "block256": _dynamic_linear_per_block_build,
        "block512": _dynamic_linear_per_block_build,
        "block1024": _dynamic_linear_per_block_build,
        "block2048": _dynamic_linear_per_block_build,
        "block4096": _dynamic_linear_per_block_build,
        "block8192": _dynamic_linear_per_block_build,
        "block2048qt": _dynamic_linear_quantized_transposed_block_build,
        "block4096qt": _dynamic_linear_quantized_transposed_block_build,
        "block8192qt": _dynamic_linear_quantized_transposed_block_build,
        "w8a16": _dynamic_linear_w8a16_build,
        "static_w8a16": _static_linear_w8a16_build,
        "fp16": _dynamic_linear_fp16_build,
    }
    if mlp_down_linear_mode is None:
        mlp_down_linear_mode = mlp_linear_mode
    matmul_builds = {"dynamic": None, "fp16": _dynamic_matmul_fp16_build}
    if attention_linear_mode not in linear_builds:
        raise ValueError(f"Unsupported attention linear mode: {attention_linear_mode}")
    if mlp_linear_mode not in linear_builds:
        raise ValueError(f"Unsupported MLP linear mode: {mlp_linear_mode}")
    if mlp_down_linear_mode not in linear_builds:
        raise ValueError(
            f"Unsupported MLP down linear mode: {mlp_down_linear_mode}"
        )
    if attention_matmul_mode not in (*matmul_builds, "fixed8x16", "fixed16"):
        raise ValueError(f"Unsupported attention matmul mode: {attention_matmul_mode}")

    def configure_linear(projection, mode):
        if mode == "dynamic":
            return
        if mode == "static_w8a16":
            _enable_static_linear_w8a16(projection)
            return
        if mode.startswith("block") and mode.endswith("qt"):
            projection._pi0_activation_block_size = int(
                mode.removeprefix("block").removesuffix("qt")
            )
        elif mode.startswith("block"):
            projection._pi0_activation_block_size = int(
                mode.removeprefix("block")
            )
        build = linear_builds[mode]
        projection.build = types.MethodType(build, projection)

    layers = paligemma.model.layers[: paligemma.model.config.num_hidden_layers]
    if attention_linear_last_layer_count is None:
        attention_linear_last_layer_count = len(layers)
    if not 0 <= attention_linear_last_layer_count <= len(layers):
        raise ValueError(
            "Attention linear last layer count must be in "
            f"[0, {len(layers)}], got {attention_linear_last_layer_count}"
        )
    attention_linear_layer_start = len(layers) - attention_linear_last_layer_count
    attention_linear_layer_indices = list(
        range(attention_linear_layer_start, len(layers))
    )
    if mlp_down_linear_layer_count is None:
        mlp_down_linear_layer_count = len(layers)
    if not 0 <= mlp_down_linear_layer_count <= len(layers):
        raise ValueError(
            "MLP down linear layer count must be in "
            f"[0, {len(layers)}], got {mlp_down_linear_layer_count}"
        )
    if not 0 <= fp16_last_layers <= len(layers):
        raise ValueError(
            f"fp16_last_layers must be in [0, {len(layers)}], "
            f"got {fp16_last_layers}"
        )
    fp16_layer_start = len(layers) - fp16_last_layers
    fp16_layer_indices = list(range(fp16_layer_start, len(layers)))
    for layer_index, layer in enumerate(layers):
        force_fp16 = fp16_last_layers > 0 and layer_index >= fp16_layer_start
        layer_attention_linear_mode = (
            "fp16"
            if force_fp16
            else (
                attention_linear_mode
                if layer_index >= attention_linear_layer_start
                else "dynamic"
            )
        )
        layer_mlp_linear_mode = "fp16" if force_fp16 else mlp_linear_mode
        layer_mlp_down_linear_mode = (
            "fp16"
            if force_fp16
            else (
                mlp_down_linear_mode
                if layer_index < mlp_down_linear_layer_count
                else mlp_linear_mode
            )
        )
        layer_attention_matmul_mode = (
            "fp16" if force_fp16 else attention_matmul_mode
        )
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        ):
            configure_linear(projection, layer_attention_linear_mode)
        if layer_attention_matmul_mode in ("fixed8x16", "fixed16"):
            if layer_attention_matmul_mode == "fixed8x16":
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=8, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=8, out_bits=None
                )
            else:
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
            layer.self_attn.qk.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.sv
            )
        elif matmul_builds.get(layer_attention_matmul_mode) is not None:
            attention_matmul_build = matmul_builds[layer_attention_matmul_mode]
            layer.self_attn.qk.build = types.MethodType(
                attention_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                attention_matmul_build, layer.self_attn.sv
            )
        for projection in (layer.mlp.gate_proj, layer.mlp.up_proj):
            configure_linear(projection, layer_mlp_linear_mode)
        configure_linear(layer.mlp.down_proj, layer_mlp_down_linear_mode)

    result = {
        "attention_linear_mode": attention_linear_mode,
        "attention_linear_last_layer_count": attention_linear_last_layer_count,
        "attention_linear_layer_indices": attention_linear_layer_indices,
        "mlp_linear_mode": mlp_linear_mode,
        "mlp_down_linear_mode": mlp_down_linear_mode,
        "mlp_down_linear_layer_count": mlp_down_linear_layer_count,
        "mlp_down_linear_layer_indices": list(
            range(mlp_down_linear_layer_count)
        ),
        "attention_matmul_mode": attention_matmul_mode,
        "fp16_last_layers": fp16_last_layers,
        "fp16_layer_indices": fp16_layer_indices,
        "layer_count": len(layers),
    }
    if attention_linear_mode.startswith("block"):
        result["attention_linear_block_size"] = int(
            attention_linear_mode.removeprefix("block")
        )
    if mlp_linear_mode.startswith("block"):
        result["mlp_linear_block_size"] = int(
            mlp_linear_mode.removeprefix("block")
        )
    if mlp_down_linear_mode.startswith("block"):
        if mlp_down_linear_mode.endswith("qt"):
            result["mlp_down_linear_block_size"] = int(
                mlp_down_linear_mode.removeprefix("block").removesuffix("qt")
            )
            result["mlp_down_linear_quantized_transposed"] = True
        else:
            result["mlp_down_linear_block_size"] = int(
                mlp_down_linear_mode.removeprefix("block")
            )
    if attention_matmul_mode == "fixed8x16":
        result["attention_matmul_qk_bits"] = [8, 16]
        result["attention_matmul_sv_bits"] = [16, 8]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    elif attention_matmul_mode == "fixed16":
        result["attention_matmul_qk_bits"] = [16, 16]
        result["attention_matmul_sv_bits"] = [16, 16]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    return result


def configure_siglip_export_precision(
    siglip,
    patch_embedding_mode: str = "quant8",
    position_embedding_mode: str = "quant8",
    attention_linear_mode: str = "dynamic",
    mlp_linear_mode: str = "dynamic",
    attention_matmul_mode: str = "dynamic",
    attention_matmul_last_layers: int | None = None,
    fp16_last_layers: int = 0,
    projector_linear_mode: str = "dynamic",
    layernorm_mode: str = "standard",
) -> dict:
    if patch_embedding_mode not in ("quant8", "fp16"):
        raise ValueError(f"Unsupported patch embedding mode: {patch_embedding_mode}")
    if position_embedding_mode not in ("quant8", "fp16"):
        raise ValueError(
            f"Unsupported position embedding mode: {position_embedding_mode}"
        )
    linear_builds = {
        "dynamic": None,
        "fakequant16": None,
        "w8a16": _dynamic_linear_w8a16_build,
        "fp16": _dynamic_linear_fp16_build,
    }
    matmul_builds = {"dynamic": None, "fp16": _dynamic_matmul_fp16_build}
    if attention_linear_mode not in linear_builds:
        raise ValueError(f"Unsupported attention linear mode: {attention_linear_mode}")
    if mlp_linear_mode not in linear_builds:
        raise ValueError(f"Unsupported MLP linear mode: {mlp_linear_mode}")
    if projector_linear_mode not in linear_builds:
        raise ValueError(
            f"Unsupported multimodal projector linear mode: {projector_linear_mode}"
        )
    if attention_matmul_mode not in (*matmul_builds, "fixed8x16", "fixed16"):
        raise ValueError(f"Unsupported attention matmul mode: {attention_matmul_mode}")

    attention_linear_build = linear_builds[attention_linear_mode]
    mlp_linear_build = linear_builds[mlp_linear_mode]
    projector_linear_build = linear_builds[projector_linear_mode]
    attention_matmul_build = matmul_builds.get(attention_matmul_mode)
    layers = siglip.model.encoder.layers
    if patch_embedding_mode == "fp16":
        patch_embedding = siglip.model.embeddings.patch_embedding
        patch_embedding.build = types.MethodType(
            _siglip_patch_embedding_fp16_build, patch_embedding
        )
    if position_embedding_mode == "fp16":
        position_embedding = siglip.model.embeddings.position_embedding
        position_embedding.build = types.MethodType(
            _siglip_position_embedding_fp16_build, position_embedding
        )
    fakequant_replacement_count = 0
    if attention_linear_mode == "fakequant16":
        for layer in layers:
            for attribute in ("q_proj", "k_proj", "v_proj", "out_proj"):
                fakequant_replacement_count += _replace_siglip_linear(
                    layer.self_attn, attribute
                )
    if mlp_linear_mode == "fakequant16":
        for layer in layers:
            for attribute in ("fc1", "fc2"):
                fakequant_replacement_count += _replace_siglip_linear(
                    layer.mlp, attribute
                )
    if projector_linear_mode == "fakequant16":
        fakequant_replacement_count += _replace_siglip_linear(
            siglip.model.multi_modal_projector, "linear", output_fp16=True
        )
    if not 0 <= fp16_last_layers <= len(layers):
        raise ValueError(
            f"fp16_last_layers must be in [0, {len(layers)}], got {fp16_last_layers}"
        )
    fp16_layer_indices = list(range(len(layers) - fp16_last_layers, len(layers)))
    fp16_layer_index_set = set(fp16_layer_indices)
    layernorm_replacement_count = _configure_siglip_layernorms(siglip, layernorm_mode)
    if attention_matmul_last_layers is None:
        attention_matmul_last_layers = len(layers)
    if not 0 <= attention_matmul_last_layers <= len(layers):
        raise ValueError(
            "attention_matmul_last_layers must be in "
            f"[0, {len(layers)}], got {attention_matmul_last_layers}"
        )
    attention_matmul_layer_indices = list(
        range(len(layers) - attention_matmul_last_layers, len(layers))
    )
    if attention_matmul_mode == "dynamic":
        attention_matmul_layer_indices = []
    attention_matmul_layer_index_set = set(attention_matmul_layer_indices)

    for layer_index, layer in enumerate(layers):
        layer_attention_linear_build = (
            _dynamic_linear_fp16_build
            if layer_index in fp16_layer_index_set
            else attention_linear_build
        )
        layer_mlp_linear_build = (
            _dynamic_linear_fp16_build
            if layer_index in fp16_layer_index_set
            else mlp_linear_build
        )
        if layer_attention_linear_build is not None:
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.out_proj,
            ):
                projection.build = types.MethodType(
                    layer_attention_linear_build, projection
                )
        if (
            layer_index in attention_matmul_layer_index_set
            and attention_matmul_mode in ("fixed8x16", "fixed16")
        ):
            if attention_matmul_mode == "fixed8x16":
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=8, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=8, out_bits=None
                )
            else:
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
            layer.self_attn.qk.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.sv
            )
        elif (
            layer_index in attention_matmul_layer_index_set
            and attention_matmul_build is not None
        ):
            layer.self_attn.qk.build = types.MethodType(
                attention_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                attention_matmul_build, layer.self_attn.sv
            )
        if layer_mlp_linear_build is not None:
            for projection in (layer.mlp.fc1, layer.mlp.fc2):
                projection.build = types.MethodType(layer_mlp_linear_build, projection)

    if projector_linear_build is not None:
        projector = siglip.model.multi_modal_projector.linear
        projector.build = types.MethodType(projector_linear_build, projector)

    result = {
        "patch_embedding_mode": patch_embedding_mode,
        "position_embedding_mode": position_embedding_mode,
        "attention_linear_mode": attention_linear_mode,
        "mlp_linear_mode": mlp_linear_mode,
        "attention_matmul_mode": attention_matmul_mode,
        "attention_matmul_last_layers": attention_matmul_last_layers,
        "attention_matmul_layer_indices": attention_matmul_layer_indices,
        "fp16_last_layers": fp16_last_layers,
        "fp16_layer_indices": fp16_layer_indices,
        "projector_linear_mode": projector_linear_mode,
        "layernorm_mode": layernorm_mode,
        "layernorm_replacement_count": layernorm_replacement_count,
        "fakequant_linear_replacement_count": fakequant_replacement_count,
        "layer_count": len(layers),
        "softmax_dtype": "float32",
        "softmax_output_dtype": "float16",
    }
    if attention_matmul_mode == "fixed8x16":
        result["attention_matmul_qk_bits"] = [8, 16]
        result["attention_matmul_sv_bits"] = [16, 8]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    elif attention_matmul_mode == "fixed16":
        result["attention_matmul_qk_bits"] = [16, 16]
        result["attention_matmul_sv_bits"] = [16, 16]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    return result


def configure_expert_export_precision(
    expert,
    transformer_linear_mode: str = "dynamic",
    projection_linear_mode: str = "dynamic",
    attention_matmul_mode: str = "dynamic",
    attention_matmul_last_layers: int | None = None,
    fp16_last_layers: int = 0,
    output_linear_mode: str = "dynamic",
) -> dict:
    linear_builds = {
        "dynamic": None,
        "w8a16": _dynamic_linear_w8a16_build,
        "fp16": _dynamic_linear_fp16_build,
    }
    matmul_builds = {
        "dynamic": None,
        "fp16": _dynamic_matmul_fp16_build,
    }
    for name, mode in (
        ("transformer linear", transformer_linear_mode),
        ("projection linear", projection_linear_mode),
        ("output linear", output_linear_mode),
    ):
        if mode not in linear_builds:
            raise ValueError(f"Unsupported {name} mode: {mode}")
    if attention_matmul_mode not in (*matmul_builds, "fixed8x16", "fixed16"):
        raise ValueError(
            f"Unsupported attention matmul mode: {attention_matmul_mode}"
        )

    layers = expert.model.model.layers
    layer_count = len(layers)
    if fp16_last_layers < 0 or fp16_last_layers > layer_count:
        raise ValueError(
            f"fp16_last_layers must be in [0, {layer_count}], got {fp16_last_layers}"
        )
    if attention_matmul_last_layers is None:
        attention_matmul_last_layers = layer_count
    if not 0 <= attention_matmul_last_layers <= layer_count:
        raise ValueError(
            "attention_matmul_last_layers must be in "
            f"[0, {layer_count}], got {attention_matmul_last_layers}"
        )
    attention_matmul_layer_indices = list(
        range(layer_count - attention_matmul_last_layers, layer_count)
    )
    if attention_matmul_mode == "dynamic":
        attention_matmul_layer_indices = []
    attention_matmul_layer_index_set = set(attention_matmul_layer_indices)

    def configure_linear(projection, mode: str) -> None:
        build = linear_builds[mode]
        if build is not None:
            projection.build = types.MethodType(build, projection)

    fp16_layer_start = layer_count - fp16_last_layers
    fp16_layer_indices = []
    for layer_index, layer in enumerate(layers):
        layer_linear_mode = transformer_linear_mode
        if fp16_last_layers and layer_index >= fp16_layer_start:
            layer_linear_mode = "fp16"
            fp16_layer_indices.append(layer_index)
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            configure_linear(projection, layer_linear_mode)

        attention_matmul_build = matmul_builds.get(attention_matmul_mode)
        if (
            layer_index in attention_matmul_layer_index_set
            and attention_matmul_mode in ("fixed8x16", "fixed16")
        ):
            if attention_matmul_mode == "fixed8x16":
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=8, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=8, out_bits=None
                )
            else:
                layer.self_attn.qk = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
                layer.self_attn.sv = FakeQuantMatmul(
                    x_bits=16, y_bits=16, out_bits=None
                )
            layer.self_attn.qk.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                _fixed_quant_transposed_matmul_build, layer.self_attn.sv
            )
        elif (
            layer_index in attention_matmul_layer_index_set
            and attention_matmul_build is not None
        ):
            layer.self_attn.qk.build = types.MethodType(
                attention_matmul_build, layer.self_attn.qk
            )
            layer.self_attn.sv.build = types.MethodType(
                attention_matmul_build, layer.self_attn.sv
            )

    projection_names = (
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    )
    for projection_name in projection_names:
        configure_linear(getattr(expert.model, projection_name), projection_linear_mode)
    configure_linear(expert.model.action_out_proj, output_linear_mode)

    result = {
        "transformer_linear_mode": transformer_linear_mode,
        "projection_linear_mode": projection_linear_mode,
        "attention_matmul_mode": attention_matmul_mode,
        "attention_matmul_last_layers": attention_matmul_last_layers,
        "attention_matmul_layer_indices": attention_matmul_layer_indices,
        "fp16_last_layers": fp16_last_layers,
        "fp16_layer_indices": fp16_layer_indices,
        "output_linear_mode": output_linear_mode,
        "projection_names": list(projection_names),
        "layer_count": layer_count,
        "softmax_dtype": "float32",
        "softmax_output_dtype": "float16",
    }
    if attention_matmul_mode == "fixed8x16":
        result["attention_matmul_qk_bits"] = [8, 16]
        result["attention_matmul_sv_bits"] = [16, 8]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    elif attention_matmul_mode == "fixed16":
        result["attention_matmul_qk_bits"] = [16, 16]
        result["attention_matmul_sv_bits"] = [16, 16]
        result["attention_matmul_accumulation_dtype"] = "float32"
        result["attention_matmul_output_dtype"] = "float16"
    return result


def _gemma_attention_forward(self, hidden_states, cos, sin, attention_mask):
    assert hidden_states.ndim == 3
    batch_size, sequence_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.reshape(
        [sequence_length, self.num_attention_heads, self.head_dim]
    ).transpose(1, 0)
    key_states = key_states.reshape(
        [sequence_length, self.num_key_value_heads, self.head_dim]
    ).transpose(1, 0)
    value_states = value_states.reshape(
        [sequence_length, self.num_key_value_heads, self.head_dim]
    ).transpose(1, 0)
    query_states, key_states = self.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    new_key = key_states
    new_value = value_states
    key_states_transposed = key_states.transpose(2, 1)
    head_count, query_length, _ = query_states.shape
    query_states = query_states.reshape(
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            self.head_dim,
        ]
    )
    attention_weights = self.qk(query_states, key_states_transposed)
    attention_weights = attention_weights.reshape(
        [head_count, sequence_length, sequence_length]
    )
    attention_weights = attention_weights * self.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_weights = attention_weights + causal_mask
    attention_weights = torch.nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attention_weights = attention_weights.reshape(
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            sequence_length,
        ]
    )
    attention_output = self.sv(attention_weights, value_states)
    attention_output = attention_output.reshape(
        [head_count, sequence_length, self.head_dim]
    ).transpose(1, 0)
    attention_output = attention_output.reshape(
        [batch_size, sequence_length, self.hidden_size]
    )
    attention_output = self.o_proj(attention_output)
    return attention_output, attention_weights, new_key, new_value


def _siglip_attention_build(self, hidden_states, output_attentions=False):
    sequence_length = hidden_states.type.shape[1]
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = leap.reshape(
        query_states, [sequence_length, self.num_heads, self.head_dim]
    )
    query_states = leap.transpose(query_states, [1, 0, 2])
    key_states = leap.reshape(
        key_states, [sequence_length, self.num_heads, self.head_dim]
    )
    key_states = leap.transpose(key_states, [1, 0, 2])
    value_states = leap.reshape(
        value_states, [sequence_length, self.num_heads, self.head_dim]
    )
    value_states = leap.transpose(value_states, [1, 2, 0])
    attention_weights = self.qk(query_states, key_states)
    attention_weights = self.mul_attn_weight(attention_weights, self.scale)
    attention_weights = leap.cast_type(
        attention_weights, output_type=leap.float32
    )
    attention_weights = leap.softmax(attention_weights, -1)
    attention_weights = leap.cast_type(
        attention_weights, output_type=leap.float16
    )
    returned_attention_weights = attention_weights
    attention_output = self.sv(attention_weights, value_states)
    attention_output = leap.transpose(attention_output, [1, 0, 2])
    attention_output = leap.reshape(
        attention_output, [sequence_length, self.embed_dim]
    )
    attention_output = self.out_proj(attention_output)
    if not output_attentions:
        returned_attention_weights = None
    return attention_output, returned_attention_weights


def _expert_attention_forward(
    self,
    hidden_states,
    attention_mask,
    cache_k,
    cache_v,
    cos,
    sin,
):
    assert hidden_states.ndim == 3
    batch_size, sequence_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.reshape(
        [sequence_length, self.num_attention_heads, self.head_dim]
    ).transpose(1, 0)
    key_states = key_states.reshape(
        [sequence_length, self.num_key_value_heads, self.head_dim]
    ).transpose(1, 0)
    value_states = value_states.reshape(
        [sequence_length, self.num_key_value_heads, self.head_dim]
    ).transpose(1, 0)
    query_states, key_states = self.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    key_states = torch.cat([cache_k, key_states], -2)
    value_states = torch.cat([cache_v, value_states], -2)
    _, context_length, _ = key_states.shape
    key_states_transposed = key_states.transpose(2, 1)
    head_count, query_length, _ = query_states.shape
    query_states = query_states.reshape(
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            self.head_dim,
        ]
    )
    attention_weights = self.qk(query_states, key_states_transposed)
    attention_weights = attention_weights.reshape(
        [head_count, sequence_length, context_length]
    )
    attention_weights = attention_weights * self.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_weights = attention_weights + causal_mask
    attention_weights = torch.nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attention_weights = attention_weights.reshape(
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            context_length,
        ]
    )
    attention_output = self.sv(attention_weights, value_states)
    attention_output = attention_output.reshape(
        [head_count, sequence_length, self.head_dim]
    ).transpose(1, 0)
    attention_output = attention_output.reshape([batch_size, sequence_length, -1])
    attention_output = self.o_proj(attention_output)
    return attention_output, attention_weights


def _gemma_expert_reference_forward(
    self,
    state,
    x_t,
    denoise_idx,
    attention_mask,
    position_ids,
    caches,
):
    output_dtype = x_t.dtype
    projection_dtype = self.state_proj.weight.dtype
    state = state.to(dtype=projection_dtype)
    x_t = x_t.to(dtype=projection_dtype)
    state_embedding = self.state_proj(state)[:, None, :]
    time_embedding = self.sinusoidal_lookup_table[denoise_idx]
    time_embedding = time_embedding.to(device=state.device, dtype=projection_dtype)
    action_embedding = self.action_in_proj(x_t)
    time_embedding = time_embedding[:, None, :].expand_as(action_embedding)
    action_time_embedding = torch.cat([action_embedding, time_embedding], dim=2)
    action_time_embedding = self.action_time_mlp_in(action_time_embedding)
    action_time_embedding = torch.nn.functional.silu(action_time_embedding)
    action_time_embedding = self.action_time_mlp_out(action_time_embedding)
    inputs_embeds = torch.cat([state_embedding, action_time_embedding], dim=1)
    inner_dtype = self.model.layers[0].self_attn.q_proj.weight.dtype
    inputs_embeds = inputs_embeds.to(dtype=inner_dtype)
    outputs_embeds = self.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        caches=caches,
    )
    suffix_output = outputs_embeds[:, -self.action_horizon :]
    suffix_output = suffix_output.to(dtype=self.action_out_proj.weight.dtype)
    suffix_output = self.action_out_proj(suffix_output)
    updated_actions = x_t - 0.1 * suffix_output
    return updated_actions.to(dtype=output_dtype)


def _gemma_attention_build(self, hidden_states, cos, sin, attention_mask):
    sequence_length = hidden_states.type.shape[1]
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = leap.reshape(
        query_states, [sequence_length, self.num_attention_heads, self.head_dim]
    )
    query_states = leap.transpose(query_states, [1, 0, 2])
    key_states = leap.reshape(
        key_states, [sequence_length, self.num_key_value_heads, self.head_dim]
    )
    key_states = leap.transpose(key_states, [1, 0, 2])
    value_states = leap.reshape(
        value_states, [sequence_length, self.num_key_value_heads, self.head_dim]
    )
    value_states = leap.transpose(value_states, [1, 0, 2])
    query_states, key_states = self.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    new_key = key_states
    new_value = value_states
    head_count, query_length, _ = query_states.type.shape
    query_states = leap.reshape(
        query_states,
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            self.head_dim,
        ],
    )
    attention_weights = self.qk(query_states, key_states)
    attention_weights = leap.reshape(
        attention_weights, [head_count, sequence_length, sequence_length]
    )
    attention_weights = leap.mul(attention_weights, self.scaling)
    attention_weights = leap.cast_type(attention_weights, output_type=leap.float32)
    if attention_mask is not None:
        if len(attention_mask.type.shape) == len(attention_weights.type.shape) - 1:
            attention_mask = leap.reshape(
                attention_mask, [1, sequence_length, sequence_length]
            )
        attention_mask = leap.cast_type(attention_mask, output_type=leap.float32)
        attention_weights = leap.add(attention_weights, attention_mask)
    attention_weights = leap.softmax(attention_weights, -1)
    attention_weights = leap.cast_type(attention_weights, output_type=leap.float16)
    attention_weights = leap.reshape(
        attention_weights,
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            sequence_length,
        ],
    )
    value_states = leap.transpose(value_states, [0, 2, 1])
    attention_output = self.sv(attention_weights, value_states)
    attention_output = leap.reshape(
        attention_output, [head_count, sequence_length, self.head_dim]
    )
    attention_output = leap.transpose(attention_output, [1, 0, 2])
    attention_output = leap.reshape(
        attention_output, [sequence_length, self.hidden_size]
    )
    attention_output = self.o_proj(attention_output)
    return attention_output, attention_weights, new_key, new_value


def _expert_attention_build(
    self,
    hidden_states,
    attention_mask,
    cache_k,
    cache_v,
    cos,
    sin,
):
    sequence_length = hidden_states.type.shape[1]
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = leap.reshape(
        query_states, [sequence_length, self.num_attention_heads, self.head_dim]
    )
    query_states = leap.transpose(query_states, [1, 0, 2])
    key_states = leap.reshape(
        key_states, [sequence_length, self.num_key_value_heads, self.head_dim]
    )
    key_states = leap.transpose(key_states, [1, 0, 2])
    value_states = leap.reshape(
        value_states, [sequence_length, self.num_key_value_heads, self.head_dim]
    )
    value_states = leap.transpose(value_states, [1, 0, 2])
    query_states, key_states = self.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    key_states = leap.concat([cache_k, key_states], 1)
    value_states = leap.concat([cache_v, value_states], 1)
    _, context_length, _ = key_states.type.shape
    head_count, query_length, _ = query_states.type.shape
    query_states = leap.reshape(
        query_states,
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            self.head_dim,
        ],
    )
    attention_weights = self.qk(query_states, key_states)
    attention_weights = leap.reshape(
        attention_weights, [head_count, sequence_length, context_length]
    )
    attention_weights = leap.mul(attention_weights, self.scaling)
    attention_weights = leap.cast_type(attention_weights, output_type=leap.float32)
    if attention_mask is not None:
        if len(attention_mask.type.shape) == len(attention_weights.type.shape) - 1:
            attention_mask = leap.reshape(
                attention_mask, [1, sequence_length, context_length]
            )
        attention_mask = leap.cast_type(attention_mask, output_type=leap.float32)
        attention_weights = leap.add(attention_weights, attention_mask)
    attention_weights = leap.softmax(attention_weights, -1)
    attention_weights = leap.cast_type(attention_weights, output_type=leap.float16)
    attention_weights = leap.reshape(
        attention_weights,
        [
            self.num_key_value_heads,
            self.num_key_value_groups * query_length,
            context_length,
        ],
    )
    value_states = leap.transpose(value_states, [0, 2, 1])
    attention_output = self.sv(attention_weights, value_states)
    attention_output = leap.reshape(
        attention_output, [head_count, sequence_length, self.head_dim]
    )
    attention_output = leap.transpose(attention_output, [1, 0, 2])
    attention_output = leap.reshape(attention_output, [sequence_length, -1])
    attention_output = self.o_proj(attention_output)
    return attention_output, attention_weights


def install_pi0_attention_precision_patch(include_leap_build: bool = False) -> dict:
    RotaryPosEmb.apply_rotary_pos_emb_torch = _apply_rotary_pos_emb_torch
    GemmaAttention.forward = _gemma_attention_forward
    ExpertAttention.forward = _expert_attention_forward
    GemmaMLP.forward = _gemma_mlp_tanh_forward
    FakeQuantGELU.forward = _fake_quant_gelu_tanh_forward
    result = {
        "torch_softmax_dtype": "float32",
        "torch_softmax_output_dtype": "query",
        "torch_rope_output_dtype": "query",
        "hidden_activation": "gelu_pytorch_tanh",
        "leap_build_patched": bool(include_leap_build),
    }
    if include_leap_build:
        GemmaAttention.build = _gemma_attention_build
        ExpertAttention.build = _expert_attention_build
        SiglipAttention.build = _siglip_attention_build
        GemmaMLP.build = _gemma_mlp_tanh_build
        FakeQuantGELU.build = _fake_quant_gelu_tanh_build
        result["leap_softmax_dtype"] = "float32"
        result["leap_softmax_output_dtype"] = "float16"
        result["siglip_leap_softmax_dtype"] = "float32"
        result["siglip_leap_softmax_output_dtype"] = "float16"
    return result


def configure_pi0_reference_precision(
    siglip,
    paligemma,
    expert,
    device: str,
    language_dtype: torch.dtype,
) -> dict:
    siglip.model.to(device=device, dtype=torch.float32)
    paligemma.model.to(device=device, dtype=language_dtype)
    expert.model.to(device=device, dtype=language_dtype)
    for module in paligemma.model.modules():
        if isinstance(module, GemmaRMSNorm):
            module.weight.data = module.weight.data.to(device=device, dtype=torch.float32)
        if isinstance(module, GemmaMLP):
            module.forward = types.MethodType(_gemma_mlp_tanh_forward, module)
    for module in expert.model.model.modules():
        if isinstance(module, GemmaRMSNorm):
            module.weight.data = module.weight.data.to(device=device, dtype=torch.float32)
        if isinstance(module, GemmaMLP):
            module.forward = types.MethodType(_gemma_mlp_tanh_forward, module)
    for module in siglip.model.modules():
        if isinstance(module, FakeQuantGELU):
            module.forward = types.MethodType(_fake_quant_gelu_tanh_forward, module)
    projection_names = (
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_out_proj",
    )
    for name in projection_names:
        getattr(expert.model, name).to(device=device, dtype=torch.float32)
    expert.model.forward = types.MethodType(
        _gemma_expert_reference_forward, expert.model
    )
    return {
        "vision_dtype": "float32",
        "language_dtype": str(language_dtype),
        "rmsnorm_weight_dtype": "float32",
        "expert_projection_dtype": "float32",
        "expert_state_dtype": "float32",
        "expert_action_dtype": "float32",
        "hidden_activation": "gelu_pytorch_tanh",
    }
