#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precision/Kernel configuration utilities.

Provides:
- configure_precision(): enable TF32 and Flash SDP if available and requested
- get_amp_settings(): read AMP enable and dtype from env (defaults to CUDA+bf16)
- make_grad_scaler(): create GradScaler when using fp16; no-op for bf16

Env toggles (all optional):
- PQC_TF32: '1' to enable TF32 matmul/cudnn (default: '1' on CUDA)
- PQC_FLASH_SDP: '1' to prefer FlashAttention SDPA kernel (default: '1' on CUDA)
- PQC_AMP: '1' to enable AMP for model forward (default: '1' on CUDA)
- PQC_AMP_DTYPE: 'bf16' (default) or 'fp16'
"""
from __future__ import annotations

import os
import torch


def _is_true(s: str | None, default: bool = False) -> bool:
    if s is None:
        return bool(default)
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def get_amp_settings() -> tuple[bool, torch.dtype | None]:
    has_cuda = torch.cuda.is_available()
    amp_enabled = _is_true(os.environ.get("PQC_AMP"), default=has_cuda)
    if not amp_enabled or not has_cuda:
        return False, None
    amp_str = str(os.environ.get("PQC_AMP_DTYPE", "bf16")).strip().lower()
    if amp_str in ("bf16", "bfloat16"):
        return True, torch.bfloat16
    if amp_str in ("fp16", "float16", "half"):
        return True, torch.float16
    # fallback
    return True, torch.bfloat16


def make_grad_scaler(amp_enabled: bool, amp_dtype: torch.dtype | None):
    """Return a GradScaler when using fp16 AMP; for bf16 returns a disabled scaler (None).
    In PyTorch, gradient scaling is beneficial for float16 but unnecessary for bfloat16.
    """
    if not (amp_enabled and torch.cuda.is_available()):
        return None
    if amp_dtype == torch.float16:
        try:
            return torch.cuda.amp.GradScaler()
        except Exception:
            return None
    return None


def configure_precision(verbose: bool = True) -> dict:
    """Enable TF32 and Flash SDP if available and requested.
    Returns a dict describing the effective configuration.
    """
    cfg = {"tf32": False, "flash_sdp": False}
    has_cuda = torch.cuda.is_available()
    # TF32
    tf32_on = _is_true(os.environ.get("PQC_TF32"), default=has_cuda)
    try:
        if tf32_on and has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            cfg["tf32"] = True
    except Exception:
        cfg["tf32"] = False
    # Flash SDP
    flash_on = _is_true(os.environ.get("PQC_FLASH_SDP"), default=has_cuda)
    if flash_on and has_cuda:
        try:
            # Preferred API (PyTorch 2.0+)
            from torch.backends.cuda import sdp_kernel  # type: ignore
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            cfg["flash_sdp"] = True
        except Exception:
            try:
                # Legacy API (may not exist in newer versions)
                torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
                cfg["flash_sdp"] = True
            except Exception:
                cfg["flash_sdp"] = False
    if verbose:
        try:
            amp_enabled, amp_dtype = get_amp_settings()
            dtype_name = str(amp_dtype).split(".")[-1] if amp_dtype is not None else None
            print(f"[Precision] TF32={cfg['tf32']} FlashSDP={cfg['flash_sdp']} AMP={amp_enabled} dtype={dtype_name}")
        except Exception:
            pass
    return cfg
