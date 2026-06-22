
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import pytest
import torch

from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.offloader.uva import UVAOffloader


class FakeTensor:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
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
        self.pin_memory_calls = 0

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
        self.pin_memory_calls += 1
        return FakeTensor(
            device=str(self._device),
            pinned=True,
            contiguous=self._contiguous,
            numel=self._numel,
            element_size=self._element_size,
            origin=self,
        )


class FakeParam:
    def __init__(self, data: FakeTensor) -> None:
        self.data = data

    @property
    def device(self) -> torch.device:
        return self.data.device


class FakeModule:
    def __init__(self, params: dict[str, FakeParam]) -> None:
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())

    def parameters(self):
        return iter(self._params.values())


@pytest.fixture(autouse=True)
def clear_loader_cache():
    yield


def test_uva_offloader_cuda_um_hints_initialization(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.offloader.uva.is_uva_available", lambda: True)
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
        False,
        raising=False,
    )
    require = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.require_cuda_um_hints_support",
        lambda device: require.append(device),
    )
    monkeypatch.setattr("vllm.model_executor.offloader.uva.torch.cuda.is_available", lambda: False)

    offloader = UVAOffloader(1024, memory_advice="cuda_um_hints")

    assert offloader.memory_advice == "cuda_um_hints"
    assert offloader.pin_memory is False
    assert require == [0]


def test_uva_offloader_cuda_um_hints_requires_uva_enabled(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.is_uva_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
        True,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="requires UVA offload"):
        UVAOffloader(1024, memory_advice="cuda_um_hints")


def test_uva_offloader_cuda_um_hints_offload_records_metadata_and_uses_system_unified_view(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.offloader.uva.is_uva_available", lambda: True)
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.require_cuda_um_hints_support",
        lambda device: None,
    )
    monkeypatch.setattr("vllm.model_executor.offloader.uva.torch.cuda.is_available", lambda: False)

    advise_calls: list[tuple[FakeTensor, int]] = []
    sentinel = FakeTensor(device="cuda:0", pinned=False, contiguous=True)

    def advise(cpu_tensor, device_id):
        advise_calls.append((cpu_tensor, device_id))

    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.advise_cuda_um_hints_for_tensor",
        advise,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.get_system_unified_cuda_view",
        lambda cpu_tensor, device_id: sentinel,
    )

    offloader = UVAOffloader(1024, memory_advice="cuda_um_hints")
    original = FakeTensor(device="cuda:0", pinned=False, contiguous=False)
    module = FakeModule({"weight": FakeParam(original)})

    offloader._maybe_offload_to_cpu(module)

    param = module._params["weight"]
    assert param.data is sentinel
    assert param._vllm_is_uva_offloaded is True
    assert param._vllm_uva_memory_advice == "cuda_um_hints"
    assert advise_calls[0][1] == 0
    assert advise_calls[0][0].device.type == "cpu"
    assert original.to_calls[0] == ("cpu", False)


def test_uva_offloader_default_path_remains_unchanged(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.offloader.uva.is_uva_available", lambda: True)
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.should_pin_memory",
        lambda: True,
    )
    sentinel = FakeTensor(device="cuda:0", pinned=False, contiguous=True)
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.get_accelerator_view_from_cpu_tensor",
        lambda cpu_tensor: sentinel,
    )

    offloader = UVAOffloader(1024, memory_advice="none")
    original = FakeTensor(device="cuda:0", pinned=False, contiguous=True)
    module = FakeModule({"weight": FakeParam(original)})

    offloader._maybe_offload_to_cpu(module)

    param = module._params["weight"]
    assert param.data is sentinel
    assert param._vllm_is_uva_offloaded is True
    assert param._vllm_uva_memory_advice == "none"
    assert original.pin_memory_calls == 1


def test_device_loading_context_reoffloads_replaced_uva_parameter(monkeypatch):
    monkeypatch.setattr(loader_utils, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        loader_utils.envs,
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY",
        False,
        raising=False,
    )

    advise_calls: list[tuple[FakeTensor, int]] = []
    sentinel = FakeTensor(device="cuda:0", pinned=False, contiguous=True)

    def advise(cpu_tensor, device_id):
        advise_calls.append((cpu_tensor, device_id))

    monkeypatch.setattr(loader_utils, "advise_cuda_um_hints_for_tensor", advise)
    monkeypatch.setattr(
        loader_utils,
        "get_system_unified_cuda_view",
        lambda cpu_tensor, device_id: sentinel,
    )

    original = FakeTensor(device="cuda:0", pinned=False, contiguous=True)
    param = FakeParam(original)
    param._vllm_is_uva_offloaded = True
    param._vllm_uva_memory_advice = "cuda_um_hints"
    module = FakeModule({"weight": param})

    with loader_utils.device_loading_context(module, torch.device("cuda", 0)):
        replacement = FakeTensor(device="cuda:0", pinned=False, contiguous=True)
        param.data = replacement
        delattr(param, "_vllm_is_uva_offloaded")

    assert original.to_calls[0] == (torch.device("cuda", 0), True)
    assert advise_calls[0][1] == 0
    assert advise_calls[0][0].origin is replacement
    assert param.data is sentinel
    assert param._vllm_is_uva_offloaded is True
    assert param._vllm_uva_memory_advice == "cuda_um_hints"
