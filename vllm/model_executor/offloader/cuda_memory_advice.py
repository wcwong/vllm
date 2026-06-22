# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Unified Memory hints support for UVA weight offload."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch

from vllm.utils.torch_utils import get_system_unified_cuda_view_from_cpu_tensor

CUDA_UM_HINTS_MIN_VERSION = 13000

CudaUMHintsCapability = Literal[
    "unsupported",
    "hints_and_hardware_coherent_mapping",
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

    required_cuda_ops = (
        "get_system_unified_cuda_view_from_cpu_tensor",
        "cuda_advise_um_hints_for_tensor",
    )
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
        capability="hints_and_hardware_coherent_mapping",
        reason="supported",
        runtime_version=runtime_version,
        driver_version=driver_version,
        attrs=dict(attrs),
        kernel_release=kernel_release,
    )


def _unsupported_reason_for_attrs(attrs: dict[str, int]) -> str | None:
    if not attrs:
        return "failed to query CUDA Unified Memory capability attributes."

    required = (
        "concurrentManagedAccess",
        "pageableMemoryAccess",
        "pageableMemoryAccessUsesHostPageTables",
    )
    missing = [name for name in required if attrs.get(name, 0) != 1]
    if not missing:
        return None

    details = ", ".join(f"{name}={attrs.get(name, 0)}" for name in required)
    return (
        "full Unified Memory support for pageable system memory is not available: "
        f"{details}."
    )


@lru_cache(maxsize=None)
def cuda_um_hints_supported(device: int) -> CudaUMHintsSupport:
    """Return whether CUDA Unified Memory hints mode is supported.

    The probe targets CUDA-capable Linux systems where ordinary non-pinned
    system memory can be accessed by GPU work through full Unified Memory
    support for pageable system memory with hardware-coherent host page-table
    access.
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
            "vLLM was built without CUDA Unified Memory hints support.",
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
        values = torch.ops._C_cuda_utils.get_cuda_um_hints_device_attributes(
            int(device)
        )
    except Exception as exc:  # pragma: no cover - defensive probe guard
        return _unsupported_support(
            f"failed to query CUDA Unified Memory capability attributes: {exc}",
            runtime_version=runtime_version,
            driver_version=driver_version,
            kernel_release=kernel_release,
        )

    attrs = {
        "concurrentManagedAccess": int(values[0]),
        "pageableMemoryAccess": int(values[1]),
        "pageableMemoryAccessUsesHostPageTables": int(values[2]),
    }
    reason = _unsupported_reason_for_attrs(attrs)
    if reason is not None:
        return _unsupported_support(
            reason,
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


def require_cuda_um_hints_support(device: int) -> None:
    """Raise RuntimeError if CUDA Unified Memory hints mode is unsupported."""

    support = cuda_um_hints_supported(device)
    if not support.supported:
        raise RuntimeError(
            "offload_memory_advice=cuda_um_hints is not supported on this "
            f"system: {support.reason}"
        )


def advise_cuda_um_hints_for_tensor(
    tensor: torch.Tensor,
    device: int,
) -> None:
    """Apply CUDA Unified Memory advice to a CPU tensor.

    Applies cudaMemAdviseSetReadMostly and cudaMemAdviseSetAccessedBy
    to the tensor's CPU backing storage.
    """

    require_cuda_um_hints_support(device)

    if tensor.device.type != "cpu":
        raise ValueError("cuda_um_hints requires a CPU tensor.")

    if tensor.is_pinned():
        raise ValueError("cuda_um_hints requires non-pinned CPU memory.")

    if not tensor.is_contiguous():
        raise ValueError("cuda_um_hints requires contiguous CPU tensors in PR 1.")

    torch.ops._C.cuda_advise_um_hints_for_tensor(tensor, device)


def get_system_unified_cuda_view(
    cpu_tensor: torch.Tensor,
    device: int,
) -> torch.Tensor:
    """Create a CUDA tensor view over ordinary non-pinned CPU memory.

    This is the cuda_um_hints feature wrapper. It validates the platform and
    tensor preconditions, then calls the low-level torch_utils wrapper around
    the C++ op.
    """

    require_cuda_um_hints_support(device)

    if cpu_tensor.device.type != "cpu":
        raise ValueError("cuda_um_hints requires a CPU tensor.")

    if cpu_tensor.is_pinned():
        raise ValueError("cuda_um_hints requires non-pinned CPU memory.")

    if not cpu_tensor.is_contiguous():
        raise ValueError("cuda_um_hints requires contiguous CPU tensors in PR 1.")

    return get_system_unified_cuda_view_from_cpu_tensor(cpu_tensor, device)
