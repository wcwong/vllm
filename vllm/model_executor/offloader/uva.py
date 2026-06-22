# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.config.offload import OffloadMemoryAdvice
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.model_executor.offloader.cuda_memory_advice import (
    advise_cuda_um_hints_for_tensor,
    get_system_unified_cuda_view,
    require_cuda_um_hints_support,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class UVAOffloader(BaseOffloader):
    """UVA CPU weight offloader.

    Default mode stores selected offloaded weights through the existing
    CUDA-visible pinned/mapped CPU memory path. cuda_um_hints mode stores
    selected offloaded weights in ordinary non-pinned system memory, applies
    CUDA Unified Memory advice, and creates CUDA-visible system-memory tensor
    views on supported full Unified Memory platforms.

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
        memory_advice: CUDA memory advice policy for UVA offloading.
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
        memory_advice: OffloadMemoryAdvice = "none",
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()
        self.memory_advice = memory_advice

        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

        if self.memory_advice == "cuda_um_hints":
            if not self.uva_offloading:
                raise RuntimeError(
                    "offload_memory_advice=cuda_um_hints requires UVA offload; "
                    "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA must not be set."
                )

            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            require_cuda_um_hints_support(device_id)
            self.pin_memory = False
        else:
            self.pin_memory = should_pin_memory()

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{name}." for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            cpu_data = p.data.to(device="cpu")
            if not cpu_data.is_contiguous():
                cpu_data = cpu_data.contiguous()

            if self.memory_advice == "cuda_um_hints":
                if cpu_data.is_pinned():
                    raise AssertionError(
                        "cuda_um_hints offload requires non-pinned CPU memory"
                    )

                device_id = device.index
                if device_id is None:
                    device_id = torch.cuda.current_device()

                advise_cuda_um_hints_for_tensor(cpu_data, device_id)
                p.data = get_system_unified_cuda_view(cpu_data, device_id)
                p._vllm_is_uva_offloaded = True
                p._vllm_uva_memory_advice = "cuda_um_hints"
            else:
                if self.pin_memory:
                    cpu_data = cpu_data.pin_memory()

                if not self.uva_offloading:
                    p.data = cpu_data
                else:
                    p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                    p._vllm_is_uva_offloaded = True
                    p._vllm_uva_memory_advice = "none"

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
