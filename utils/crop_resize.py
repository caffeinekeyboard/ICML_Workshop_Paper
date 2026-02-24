"""
Utility functions for cropping and resizing tensors.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def _normalize_output_size(output_size: Iterable[int]) -> Tuple[int, int]:
    size = tuple(int(v) for v in output_size)
    if len(size) != 2:
        raise ValueError("output_size must be a tuple/list of (height, width)")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("output_size values must be positive")
    return size[0], size[1]


def crop_resize_horizontal(
    tensor: torch.Tensor,
    output_size: Iterable[int],
    mode: str = "bicubic",
    align_corners: bool | None = False,
    antialias: bool = True,
) -> torch.Tensor:
    """
    Crop along the height (horizontal line) to match the target aspect ratio,
    then resize to the desired output size.

    Args:
        tensor: Input tensor with shape (H, W), (C, H, W), or (N, C, H, W).
        output_size: Desired output size (out_height, out_width).
        mode: Interpolation mode used by torch.nn.functional.interpolate.
        align_corners: Align corners option for interpolation.
        antialias: Whether to apply antialiasing during interpolation.

    Returns:
        Tensor resized to output_size with the same leading dimensions
        as the input tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    out_h, out_w = _normalize_output_size(output_size)

    if tensor.dim() == 2:
        tensor_in = tensor.unsqueeze(0).unsqueeze(0)
        squeeze_dims = (0, 0)
    elif tensor.dim() == 3:
        tensor_in = tensor.unsqueeze(0)
        squeeze_dims = (0,)
    elif tensor.dim() == 4:
        tensor_in = tensor
        squeeze_dims = ()
    else:
        raise ValueError("tensor must have 2, 3, or 4 dimensions")

    in_h = tensor_in.shape[-2]
    in_w = tensor_in.shape[-1]

    target_ratio = out_w / out_h
    crop_h = int(round(in_w / target_ratio))

    if crop_h > in_h:
        crop_h = in_h

    top = max((in_h - crop_h) // 2, 0)
    bottom = top + crop_h

    cropped = tensor_in[..., top:bottom, :]

    resized = F.interpolate(
        cropped,
        size=(out_h, out_w),
        mode=mode,
        align_corners=align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        antialias=antialias if mode in {"bilinear", "bicubic"} else False,
    )

    if squeeze_dims:
        for dim in sorted(squeeze_dims, reverse=True):
            resized = resized.squeeze(dim)

    return resized
