# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.offloader import cuda_memory_advice


class FakeTensor:
    def __init__(
        self,
        *,
        device: str = "cpu",
        pinned: bool = False,
        contiguous: bool = True,
        numel: int = 4,
        element_size: int = 2,
        origin: "FakeTensor | None" = None,
    ) -> None:
        self._device = torch.device(device)
        self._pinned = pinned
        self._contiguous = contiguous
        self._numel = numel
        self._element_size = element_size
        self.origin = origin or self
        self.to_calls: list[tuple[object | None, bool]] = []

    @property
    def device(self) -> torch.device:
        return self._device

    def is_pinned(self) -> bool:
        return self._pinned

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size

    def to(self, device=None, copy: bool = False):
        self.to_calls.append((device, copy))
        if device is None:
            device = self._device
        return FakeTensor(
            device=str(torch.device(device)),
            pinned=False,
            contiguous=self._contiguous,
            numel=self._numel,
            element_size=self._element_size,
            origin=self,
        )

    def contiguous(self):
        return FakeTensor(
            device=str(self._device),
            pinned=self._pinned,
            contiguous=True,
            numel=self._numel,
            element_size=self._element_size,
            origin=self,
        )

    def pin_memory(self):
        return FakeTensor(
            device=str(self._device),
            pinned=True,
            contiguous=self._contiguous,
            numel=self._numel,
            element_size=self._element_size,
            origin=self,
        )


@pytest.fixture(autouse=True)
def clear_cache():
    cuda_memory_advice.cuda_um_hints_supported.cache_clear()
    yield
    cuda_memory_advice.cuda_um_hints_supported.cache_clear()


def _prime_supported_probe(monkeypatch) -> None:
    monkeypatch.setattr(cuda_memory_advice.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        cuda_memory_advice.platform, "release", lambda: "6.8.0-test"
    )
    monkeypatch.setattr(
        cuda_memory_advice.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        cuda_memory_advice,
        "_cuda_um_hints_supported_ops_available",
        lambda: True,
    )
    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C_cuda_utils,
        "get_cuda_runtime_version",
        lambda: 13000,
        raising=False,
    )
    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C_cuda_utils,
        "get_cuda_driver_version",
        lambda: 13000,
        raising=False,
    )
    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C_cuda_utils,
        "get_cuda_um_hints_device_attributes",
        lambda device: [0, 0, 0],
        raising=False,
    )
    monkeypatch.setattr(
        cuda_memory_advice,
        "_managed_memory_canary",
        lambda device: (True, "supported"),
    )


def test_cuda_um_hints_supported_reports_diagnostics_and_caches(monkeypatch):
    calls = {"attrs": 0}
    _prime_supported_probe(monkeypatch)

    def get_attrs(device: int):
        calls["attrs"] += 1
        assert device == 0
        return [0, 0, 0]

    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C_cuda_utils,
        "get_cuda_um_hints_device_attributes",
        get_attrs,
        raising=False,
    )

    support = cuda_memory_advice.cuda_um_hints_supported(0)

    assert support.supported
    assert support.capability == "managed_memory_hints"
    assert support.runtime_version == 13000
    assert support.driver_version == 13000
    assert support.attrs == {
        "concurrentManagedAccess": 0,
        "pageableMemoryAccess": 0,
        "pageableMemoryAccessUsesHostPageTables": 0,
    }
    assert support.reason == "supported"
    assert calls["attrs"] == 1

    cached = cuda_memory_advice.cuda_um_hints_supported(0)
    assert cached is support
    assert calls["attrs"] == 1


@pytest.mark.parametrize(
    ("mutator", "reason_fragment"),
    [
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.torch.cuda, "is_available", lambda: False
            ),
            "CUDA unavailable",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.platform, "system", lambda: "Darwin"
            ),
            "requires Linux",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.platform,
                "release",
                lambda: "5.15.0-microsoft-standard-WSL2",
            ),
            "WSL",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod, "_cuda_um_hints_supported_ops_available", lambda: False
            ),
            "built without CUDA managed-memory hints support",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.torch.ops._C_cuda_utils,
                "get_cuda_runtime_version",
                lambda: 12000,
                raising=False,
            ),
            "CUDA runtime < 13000",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.torch.ops._C_cuda_utils,
                "get_cuda_driver_version",
                lambda: 12000,
                raising=False,
            ),
            "CUDA driver < 13000",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod.torch.ops._C_cuda_utils,
                "get_cuda_um_hints_device_attributes",
                lambda device: (_ for _ in ()).throw(RuntimeError("attrs failed")),
                raising=False,
            ),
            "failed to query CUDA UM hints device attributes",
        ),
        (
            lambda mod, monkeypatch: monkeypatch.setattr(
                mod,
                "_managed_memory_canary",
                lambda device: (False, "managed canary failed"),
            ),
            "managed canary failed",
        ),
    ],
)
def test_cuda_um_hints_supported_unsupported_cases(
    monkeypatch, mutator, reason_fragment
):
    _prime_supported_probe(monkeypatch)
    mutator(cuda_memory_advice, monkeypatch)

    support = cuda_memory_advice.cuda_um_hints_supported(0)

    assert not support.supported
    assert reason_fragment in support.reason


def test_copy_tensor_to_cuda_um_hints_storage_preconditions(monkeypatch):
    sentinel = object()
    copy = Mock(return_value=sentinel)
    monkeypatch.setattr(
        cuda_memory_advice,
        "require_cuda_um_hints_support",
        lambda device: None,
    )
    monkeypatch.setattr(
        cuda_memory_advice,
        "copy_to_managed_cuda_tensor",
        copy,
    )

    valid = FakeTensor(device="cpu")
    assert cuda_memory_advice.copy_tensor_to_cuda_um_hints_storage(valid, 0) is sentinel
    copy.assert_called_once_with(valid, 0)

    with pytest.raises(ValueError, match="CPU or CUDA tensor"):
        cuda_memory_advice.copy_tensor_to_cuda_um_hints_storage(
            FakeTensor(device="meta"),
            0,
        )

    with pytest.raises(ValueError, match="contiguous"):
        cuda_memory_advice.copy_tensor_to_cuda_um_hints_storage(
            FakeTensor(device="cpu", contiguous=False),
            0,
        )

    zero_size = FakeTensor(device="cpu", numel=0)
    assert cuda_memory_advice.copy_tensor_to_cuda_um_hints_storage(zero_size, 0) is sentinel
    assert copy.call_count == 2


def test_copy_to_managed_cuda_tensor_roundtrip():
    if not hasattr(torch.ops, "_C"):
        pytest.skip("CUDA stable ops are unavailable")
    if not hasattr(torch.ops._C, "copy_to_managed_cuda_tensor"):
        pytest.skip("CUDA managed-memory op is unavailable")

    support = cuda_memory_advice.cuda_um_hints_supported(0)
    if not support.supported:
        pytest.skip(
            "requires supported CUDA managed-memory platform: "
            f"{support.reason}"
        )

    cases = (
        torch.arange(1024, dtype=torch.float32, device="cpu"),
        torch.arange(1024, dtype=torch.float16, device="cpu") / 1024,
    )

    for cpu in cases:
        managed = torch.ops._C.copy_to_managed_cuda_tensor(cpu, 0)

        assert managed.is_cuda
        assert managed.device.index == 0
        assert managed.shape == cpu.shape
        assert managed.dtype == cpu.dtype
        assert managed.stride() == cpu.stride()
        torch.testing.assert_close(managed.cpu(), cpu)

        gpu_value = (managed * 2).sum(dtype=torch.float64)
        torch.cuda.synchronize(0)
        torch.testing.assert_close(
            gpu_value.cpu(), (cpu * 2).sum(dtype=torch.float64)
        )


def test_copy_to_managed_cuda_tensor_restores_current_device():
    if not hasattr(torch.ops, "_C"):
        pytest.skip("CUDA stable ops are unavailable")
    if not hasattr(torch.ops._C, "copy_to_managed_cuda_tensor"):
        pytest.skip("CUDA managed-memory op is unavailable")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    original_device = torch.cuda.current_device()
    target_device = 1 if original_device == 0 else 0

    support = cuda_memory_advice.cuda_um_hints_supported(target_device)
    if not support.supported:
        pytest.skip(
            "requires supported CUDA managed-memory platform: "
            f"{support.reason}"
        )

    cpu = torch.arange(16, dtype=torch.float32, device="cpu")
    try:
        managed = torch.ops._C.copy_to_managed_cuda_tensor(cpu, target_device)
        assert managed.device.index == target_device
        assert torch.cuda.current_device() == original_device
    finally:
        torch.cuda.set_device(original_device)
