# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA managed-memory hints support for UVA weight offload."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch

from vllm.utils.torch_utils import copy_to_managed_cuda_tensor

CUDA_UM_HINTS_MIN_VERSION = 13000

CudaUMHintsCapability = Literal[
    "unsupported",
    "managed_memory_hints",
]


@dataclass(frozen=True)
class CudaUMHintsSupport:
    supported: bool
    capability: CudaUMHintsCapability
    reason: str
    runtime_version: int | None
    driver_version: int | None
    attrs: dict[str, int]
    kernel_release: str


def _cuda_um_hints_supported_ops_available() -> bool:
    try:
        cuda_ops = torch.ops._C
        cuda_utils_ops = torch.ops._C_cuda_utils
    except AttributeError:
        return False

    required_cuda_ops = ("copy_to_managed_cuda_tensor",)
    required_cuda_utils_ops = (
        "get_cuda_runtime_version",
        "get_cuda_driver_version",
        "get_cuda_um_hints_device_attributes",
    )
    return all(hasattr(cuda_ops, op) for op in required_cuda_ops) and all(
        hasattr(cuda_utils_ops, op) for op in required_cuda_utils_ops
    )


def _unsupported_support(
    reason: str,
    *,
    runtime_version: int | None = None,
    driver_version: int | None = None,
    attrs: dict[str, int] | None = None,
    kernel_release: str | None = None,
) -> CudaUMHintsSupport:
    return CudaUMHintsSupport(
        supported=False,
        capability="unsupported",
        reason=reason,
        runtime_version=runtime_version,
        driver_version=driver_version,
        attrs={} if attrs is None else dict(attrs),
        kernel_release=platform.release() if kernel_release is None else kernel_release,
    )


def _supported_support(
    *,
    runtime_version: int,
    driver_version: int,
    attrs: dict[str, int],
    kernel_release: str,
) -> CudaUMHintsSupport:
    return CudaUMHintsSupport(
        supported=True,
        capability="managed_memory_hints",
        reason="supported",
        runtime_version=runtime_version,
        driver_version=driver_version,
        attrs=dict(attrs),
        kernel_release=kernel_release,
    )


def _query_cuda_um_hints_device_attributes(device: int) -> dict[str, int]:
    try:
        values = torch.ops._C_cuda_utils.get_cuda_um_hints_device_attributes(int(device))
    except Exception as exc:
        raise RuntimeError(
            "failed to query CUDA UM hints device attributes: "
            f"{exc}"
        ) from exc

    return {
        "concurrentManagedAccess": int(values[0]),
        "pageableMemoryAccess": int(values[1]),
        "pageableMemoryAccessUsesHostPageTables": int(values[2]),
    }


@lru_cache(maxsize=None)
def cuda_um_hints_supported(device: int) -> CudaUMHintsSupport:
    """Return whether CUDA managed-memory hints mode is supported.

    The probe targets CUDA-capable Linux systems where vLLM can allocate
    CUDA managed memory, apply Unified Memory advice to the managed range,
    expose it as a CUDA tensor, and read it correctly from GPU work.
    """

    kernel_release = platform.release()
    if platform.system() != "Linux":
        return _unsupported_support(
            "cuda_um_hints requires Linux.",
            kernel_release=kernel_release,
        )

    lower_kernel = kernel_release.lower()
    if "microsoft" in lower_kernel or "wsl" in lower_kernel:
        return _unsupported_support(
            "cuda_um_hints is not supported on WSL.",
            kernel_release=kernel_release,
        )

    if not torch.cuda.is_available():
        return _unsupported_support(
            "CUDA unavailable.",
            kernel_release=kernel_release,
        )

    if not _cuda_um_hints_supported_ops_available():
        return _unsupported_support(
            "vLLM was built without CUDA managed-memory hints support.",
            kernel_release=kernel_release,
        )

    try:
        runtime_version = int(torch.ops._C_cuda_utils.get_cuda_runtime_version())
        driver_version = int(torch.ops._C_cuda_utils.get_cuda_driver_version())
    except Exception as exc:  # pragma: no cover - defensive probe guard
        return _unsupported_support(
            f"failed to query CUDA runtime or driver version: {exc}",
            kernel_release=kernel_release,
        )

    if runtime_version < CUDA_UM_HINTS_MIN_VERSION:
        return _unsupported_support(
            "CUDA runtime < 13000, required >= 13000. PR 1 requires CUDA 13.0+.",
            runtime_version=runtime_version,
            driver_version=driver_version,
            kernel_release=kernel_release,
        )

    if driver_version < CUDA_UM_HINTS_MIN_VERSION:
        return _unsupported_support(
            "CUDA driver < 13000, required >= 13000.",
            runtime_version=runtime_version,
            driver_version=driver_version,
            kernel_release=kernel_release,
        )

    try:
        attrs = _query_cuda_um_hints_device_attributes(int(device))
    except RuntimeError as exc:
        return _unsupported_support(
            str(exc),
            runtime_version=runtime_version,
            driver_version=driver_version,
            kernel_release=kernel_release,
        )

    canary_supported, canary_reason = _managed_memory_canary(int(device))
    if not canary_supported:
        return _unsupported_support(
            canary_reason,
            runtime_version=runtime_version,
            driver_version=driver_version,
            attrs=attrs,
            kernel_release=kernel_release,
        )

    return _supported_support(
        runtime_version=runtime_version,
        driver_version=driver_version,
        attrs=attrs,
        kernel_release=kernel_release,
    )


def _managed_memory_canary(device: int) -> tuple[bool, str]:
    try:
        cpu = torch.arange(1024, dtype=torch.float32, device="cpu")
        managed = copy_to_managed_cuda_tensor(cpu, device)

        if not managed.is_cuda:
            return False, "managed-memory op did not return a CUDA tensor"

        if managed.device.index != device:
            return False, (
                f"managed-memory op returned unexpected device: {managed.device}"
            )

        if managed.shape != cpu.shape:
            return False, "managed-memory op returned unexpected shape"

        torch.testing.assert_close(managed.cpu(), cpu)

        out = (managed * 2).sum()
        torch.cuda.synchronize(device)
        expected = (cpu * 2).sum()
        torch.testing.assert_close(out.cpu(), expected)

        return True, "supported"
    except Exception as exc:  # pragma: no cover - defensive probe guard
        return False, f"CUDA managed-memory tensor canary failed: {exc}"


def require_cuda_um_hints_support(device: int) -> None:
    """Raise RuntimeError if CUDA managed-memory hints mode is unsupported."""

    support = cuda_um_hints_supported(device)
    if not support.supported:
        raise RuntimeError(
            "offload_memory_advice=cuda_um_hints is not supported on this "
            f"system: {support.reason}"
        )


def copy_tensor_to_cuda_um_hints_storage(
    tensor: torch.Tensor,
    device: int,
) -> torch.Tensor:
    """Copy a contiguous tensor into CUDA managed storage with UM advice."""

    require_cuda_um_hints_support(device)

    if tensor.device.type not in {"cpu", "cuda"}:
        raise ValueError("cuda_um_hints requires a CPU or CUDA tensor.")

    if not tensor.is_contiguous():
        raise ValueError("cuda_um_hints requires contiguous tensors in PR 1.")

    return copy_to_managed_cuda_tensor(tensor, device)
