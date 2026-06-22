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
        lambda device: [1, 1, 1],
        raising=False,
    )


def test_cuda_um_hints_supported_uses_batched_attribute_helper(monkeypatch):
    calls = {"attrs": 0}
    _prime_supported_probe(monkeypatch)

    def get_attrs(device: int):
        calls["attrs"] += 1
        assert device == 0
        return [1, 1, 1]

    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C_cuda_utils,
        "get_cuda_um_hints_device_attributes",
        get_attrs,
        raising=False,
    )

    support = cuda_memory_advice.cuda_um_hints_supported(0)

    assert support.supported
    assert support.capability == "hints_and_hardware_coherent_mapping"
    assert support.runtime_version == 13000
    assert support.driver_version == 13000
    assert support.attrs == {
        "concurrentManagedAccess": 1,
        "pageableMemoryAccess": 1,
        "pageableMemoryAccessUsesHostPageTables": 1,
    }
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
            "built without CUDA Unified Memory hints support",
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
                lambda device: [1, 0, 1],
                raising=False,
            ),
            "full Unified Memory support",
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


def test_advise_cuda_um_hints_for_tensor_preconditions(monkeypatch):
    advise = Mock()
    monkeypatch.setattr(
        cuda_memory_advice,
        "require_cuda_um_hints_support",
        lambda device: None,
    )
    monkeypatch.setattr(
        cuda_memory_advice.torch.ops._C,
        "cuda_advise_um_hints_for_tensor",
        advise,
        raising=False,
    )

    valid = FakeTensor(device="cpu")
    cuda_memory_advice.advise_cuda_um_hints_for_tensor(valid, 0)
    advise.assert_called_once_with(valid, 0)

    with pytest.raises(ValueError, match="CPU tensor"):
        cuda_memory_advice.advise_cuda_um_hints_for_tensor(
            FakeTensor(device="cuda:0"),
            0,
        )

    with pytest.raises(ValueError, match="non-pinned"):
        cuda_memory_advice.advise_cuda_um_hints_for_tensor(
            FakeTensor(device="cpu", pinned=True),
            0,
        )

    with pytest.raises(ValueError, match="contiguous"):
        cuda_memory_advice.advise_cuda_um_hints_for_tensor(
            FakeTensor(device="cpu", contiguous=False),
            0,
        )

    zero_size = FakeTensor(device="cpu", numel=0)
    cuda_memory_advice.advise_cuda_um_hints_for_tensor(zero_size, 0)
    assert advise.call_count == 2


def test_get_system_unified_cuda_view_policy_wrapper(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        cuda_memory_advice,
        "require_cuda_um_hints_support",
        lambda device: None,
    )
    monkeypatch.setattr(
        cuda_memory_advice,
        "get_system_unified_cuda_view_from_cpu_tensor",
        lambda cpu_tensor, device: sentinel,
    )

    valid = FakeTensor(device="cpu")
    assert cuda_memory_advice.get_system_unified_cuda_view(valid, 0) is sentinel

    with pytest.raises(ValueError, match="CPU tensor"):
        cuda_memory_advice.get_system_unified_cuda_view(
            FakeTensor(device="cuda:0"),
            0,
        )

    with pytest.raises(ValueError, match="non-pinned"):
        cuda_memory_advice.get_system_unified_cuda_view(
            FakeTensor(device="cpu", pinned=True),
            0,
        )

    with pytest.raises(ValueError, match="contiguous"):
        cuda_memory_advice.get_system_unified_cuda_view(
            FakeTensor(device="cpu", contiguous=False),
            0,
        )

    zero_size = FakeTensor(device="cpu", numel=0)
    assert cuda_memory_advice.get_system_unified_cuda_view(zero_size, 0) is sentinel


def test_cuda_um_hints_advice_and_view_roundtrip():
    if not hasattr(torch.ops, "_C"):
        pytest.skip("CUDA stable ops are unavailable")
    if not hasattr(torch.ops._C, "cuda_advise_um_hints_for_tensor"):
        pytest.skip("CUDA UM hints ops are unavailable")
    if not hasattr(torch.ops._C, "get_system_unified_cuda_view_from_cpu_tensor"):
        pytest.skip("CUDA UM hints ops are unavailable")

    support = cuda_memory_advice.cuda_um_hints_supported(0)
    if not support.supported:
        pytest.skip(
            "requires supported CUDA full Unified Memory platform: "
            f"{support.reason}"
        )

    cpu = torch.arange(1024, dtype=torch.float16, device="cpu")
    if not cpu.is_contiguous():
        cpu = cpu.contiguous()

    assert not cpu.is_pinned()

    torch.ops._C.cuda_advise_um_hints_for_tensor(cpu, 0)
    gpu_view = torch.ops._C.get_system_unified_cuda_view_from_cpu_tensor(cpu, 0)

    assert gpu_view.is_cuda
    assert gpu_view.shape == cpu.shape
    assert gpu_view.stride() == cpu.stride()
    torch.testing.assert_close(gpu_view.cpu(), cpu)


def test_get_system_unified_cuda_view_opcheck():
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck is unavailable")
    if not hasattr(torch.ops, "_C"):
        pytest.skip("CUDA stable ops are unavailable")
    if not hasattr(torch.ops._C, "get_system_unified_cuda_view_from_cpu_tensor"):
        pytest.skip("CUDA UM hints ops are unavailable")

    support = cuda_memory_advice.cuda_um_hints_supported(0)
    if not support.supported:
        pytest.skip(
            "requires supported CUDA full Unified Memory platform: "
            f"{support.reason}"
        )

    result = torch.library.opcheck(
        torch.ops._C.get_system_unified_cuda_view_from_cpu_tensor,
        (torch.empty((2, 4), dtype=torch.float16), 0),
        test_utils=("test_schema", "test_faketensor"),
        raise_exception=False,
    )

    if result:
        pytest.skip(
            "opcheck is not supported for the stable-ABI view op: "
            f"{result}"
        )
