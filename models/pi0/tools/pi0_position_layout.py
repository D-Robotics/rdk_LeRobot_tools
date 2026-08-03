#!/usr/bin/env python3
from __future__ import annotations

import os

import torch

VISION_TOKENS_PER_CAMERA = 256
PHYSICAL_CAMERA_SLOTS = 3
VALID_CAMERA_SLOTS = int(os.environ.get("PI0_VALID_CAMERA_SLOTS", "1"))
if not 1 <= VALID_CAMERA_SLOTS <= PHYSICAL_CAMERA_SLOTS:
    raise ValueError(
        f"PI0_VALID_CAMERA_SLOTS must be in [1, {PHYSICAL_CAMERA_SLOTS}], "
        f"got {VALID_CAMERA_SLOTS}"
    )
LANGUAGE_TOKEN_CAPACITY = 48
ACTION_HORIZON = 50
SUFFIX_TOKEN_COUNT = ACTION_HORIZON + 1

PHYSICAL_VISION_TOKEN_COUNT = VISION_TOKENS_PER_CAMERA * PHYSICAL_CAMERA_SLOTS
VALID_VISION_TOKEN_COUNT = VISION_TOKENS_PER_CAMERA * VALID_CAMERA_SLOTS
PREFIX_TOKEN_COUNT = PHYSICAL_VISION_TOKEN_COUNT + LANGUAGE_TOKEN_CAPACITY


def compact_prefix_token_count(valid_prompt_length: int) -> int:
    valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
    return VALID_VISION_TOKEN_COUNT + valid_prompt_length

def validate_valid_prompt_length(valid_prompt_length: int) -> int:
    valid_prompt_length = int(valid_prompt_length)
    if not 1 <= valid_prompt_length <= LANGUAGE_TOKEN_CAPACITY:
        raise ValueError(
            f"valid_prompt_length must be in [1, {LANGUAGE_TOKEN_CAPACITY}], "
            f"got {valid_prompt_length}"
        )
    return valid_prompt_length


def build_paligemma_position_ids(
    valid_prompt_length: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
    front = torch.arange(
        VALID_VISION_TOKEN_COUNT,
        dtype=torch.int64,
        device=device,
    )
    empty = torch.full(
        (PHYSICAL_VISION_TOKEN_COUNT - VALID_VISION_TOKEN_COUNT,),
        VALID_VISION_TOKEN_COUNT - 1,
        dtype=torch.int64,
        device=device,
    )
    valid_language = torch.arange(
        VALID_VISION_TOKEN_COUNT,
        VALID_VISION_TOKEN_COUNT + valid_prompt_length,
        dtype=torch.int64,
        device=device,
    )
    padded_language = torch.full(
        (LANGUAGE_TOKEN_CAPACITY - valid_prompt_length,),
        VALID_VISION_TOKEN_COUNT + valid_prompt_length - 1,
        dtype=torch.int64,
        device=device,
    )
    position_ids = torch.cat([front, empty, valid_language, padded_language])
    if position_ids.numel() != PREFIX_TOKEN_COUNT:
        raise RuntimeError(f"Unexpected PaliGemma position count: {position_ids.numel()}")
    return position_ids


def build_compact_paligemma_position_ids(
    valid_prompt_length: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return torch.arange(
        compact_prefix_token_count(valid_prompt_length),
        dtype=torch.int64,
        device=device,
    )


def apply_paligemma_position_layout(model, valid_prompt_length: int) -> dict:
    position_ids = build_paligemma_position_ids(valid_prompt_length, model.cos.device)
    maximum_position = int(position_ids.max().item())
    if model.cos.shape[0] <= maximum_position or model.sin.shape[0] <= maximum_position:
        raise ValueError(
            "PaliGemma RoPE cache is too short for compressed Pi0 positions: "
            f"cos={tuple(model.cos.shape)}, sin={tuple(model.sin.shape)}, "
            f"max_position={maximum_position}"
        )
    model.cos = model.cos.index_select(0, position_ids).contiguous()
    model.sin = model.sin.index_select(0, position_ids).contiguous()
    return position_layout_manifest(valid_prompt_length)


def apply_compact_paligemma_position_layout(
    model, valid_prompt_length: int
) -> dict:
    position_ids = build_compact_paligemma_position_ids(
        valid_prompt_length, model.cos.device
    )
    maximum_position = int(position_ids.max().item())
    if model.cos.shape[0] <= maximum_position or model.sin.shape[0] <= maximum_position:
        raise ValueError(
            "PaliGemma RoPE cache is too short for compact Pi0 positions: "
            f"cos={tuple(model.cos.shape)}, sin={tuple(model.sin.shape)}, "
            f"max_position={maximum_position}"
        )
    model.cos = model.cos.index_select(0, position_ids).contiguous()
    model.sin = model.sin.index_select(0, position_ids).contiguous()
    return compact_position_layout_manifest(valid_prompt_length)


def build_prefix_valid_mask(
    valid_prompt_length: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
    valid = torch.zeros(PREFIX_TOKEN_COUNT, dtype=torch.bool, device=device)
    valid[:VALID_VISION_TOKEN_COUNT] = True
    prompt_start = PHYSICAL_VISION_TOKEN_COUNT
    valid[prompt_start : prompt_start + valid_prompt_length] = True
    return valid


def build_paligemma_attention_mask(
    valid_prompt_length: int,
    neg_value: float = -32767.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    valid = build_prefix_valid_mask(valid_prompt_length, device=device)
    visible = valid[:, None] & valid[None, :]
    mask = torch.full(
        (PREFIX_TOKEN_COUNT, PREFIX_TOKEN_COUNT),
        neg_value,
        dtype=dtype,
        device=device,
    )
    mask.masked_fill_(visible, 0.0)
    return mask.view(1, 1, PREFIX_TOKEN_COUNT, PREFIX_TOKEN_COUNT)


def build_compact_paligemma_attention_mask(
    valid_prompt_length: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    prefix_token_count = compact_prefix_token_count(valid_prompt_length)
    return torch.zeros(
        (1, 1, prefix_token_count, prefix_token_count),
        dtype=dtype,
        device=device,
    )


def build_expert_attention_mask(
    valid_prompt_length: int,
    neg_value: float = -32767.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    valid_prefix = build_prefix_valid_mask(valid_prompt_length, device=device)
    total_columns = PREFIX_TOKEN_COUNT + SUFFIX_TOKEN_COUNT
    mask = torch.zeros(
        (1, 1, SUFFIX_TOKEN_COUNT, total_columns),
        dtype=dtype,
        device=device,
    )
    invalid_prefix = (~valid_prefix).view(1, 1, 1, PREFIX_TOKEN_COUNT)
    mask[:, :, :, :PREFIX_TOKEN_COUNT].masked_fill_(invalid_prefix, neg_value)
    first_action_column = PREFIX_TOKEN_COUNT + 1
    mask[:, :, 0, first_action_column:] = neg_value
    return mask


def build_compact_expert_attention_mask(
    valid_prompt_length: int,
    neg_value: float = -32767.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    prefix_token_count = compact_prefix_token_count(valid_prompt_length)
    total_columns = prefix_token_count + SUFFIX_TOKEN_COUNT
    mask = torch.zeros(
        (1, 1, SUFFIX_TOKEN_COUNT, total_columns),
        dtype=dtype,
        device=device,
    )
    first_action_column = prefix_token_count + 1
    mask[:, :, 0, first_action_column:] = neg_value
    return mask


def build_expert_position_ids(
    valid_prompt_length: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
    start = VALID_VISION_TOKEN_COUNT + valid_prompt_length
    return torch.arange(
        start,
        start + SUFFIX_TOKEN_COUNT,
        dtype=torch.int32,
        device=device,
    ).view(1, SUFFIX_TOKEN_COUNT)


def position_layout_manifest(valid_prompt_length: int | None = None) -> dict:
    result = {
        "physical_camera_slots": PHYSICAL_CAMERA_SLOTS,
        "valid_camera_slots": VALID_CAMERA_SLOTS,
        "vision_tokens_per_camera": VISION_TOKENS_PER_CAMERA,
        "physical_vision_tokens": PHYSICAL_VISION_TOKEN_COUNT,
        "valid_vision_tokens": VALID_VISION_TOKEN_COUNT,
        "language_capacity": LANGUAGE_TOKEN_CAPACITY,
        "physical_prefix_tokens": PREFIX_TOKEN_COUNT,
        "paligemma_positions": {
            "front": [0, VALID_VISION_TOKEN_COUNT - 1],
            "empty_cameras": VALID_VISION_TOKEN_COUNT - 1,
            "language": [
                VALID_VISION_TOKEN_COUNT,
                VALID_VISION_TOKEN_COUNT + LANGUAGE_TOKEN_CAPACITY - 1,
            ],
        },
    }
    if valid_prompt_length is not None:
        valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
        expert_start = VALID_VISION_TOKEN_COUNT + valid_prompt_length
        result["valid_prompt_tokens"] = valid_prompt_length
        result["logical_prefix_tokens"] = expert_start
        result["expert_positions"] = {
            "state": expert_start,
            "actions": [expert_start + 1, expert_start + ACTION_HORIZON],
        }
    return result


def compact_position_layout_manifest(valid_prompt_length: int) -> dict:
    valid_prompt_length = validate_valid_prompt_length(valid_prompt_length)
    expert_start = compact_prefix_token_count(valid_prompt_length)
    return {
        "physical_camera_slots": VALID_CAMERA_SLOTS,
        "valid_camera_slots": VALID_CAMERA_SLOTS,
        "vision_tokens_per_camera": VISION_TOKENS_PER_CAMERA,
        "physical_vision_tokens": VALID_VISION_TOKEN_COUNT,
        "valid_vision_tokens": VALID_VISION_TOKEN_COUNT,
        "language_capacity": valid_prompt_length,
        "physical_prefix_tokens": expert_start,
        "valid_prompt_tokens": valid_prompt_length,
        "logical_prefix_tokens": expert_start,
        "paligemma_positions": {
            "front": [0, VALID_VISION_TOKEN_COUNT - 1],
            "language": [VALID_VISION_TOKEN_COUNT, expert_start - 1],
        },
        "expert_positions": {
            "state": expert_start,
            "actions": [expert_start + 1, expert_start + ACTION_HORIZON],
        },
    }
