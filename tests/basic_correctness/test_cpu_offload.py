# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ..utils import compare_two_settings


@pytest.mark.parametrize("disable_pin_memory", [False, True])
@pytest.mark.parametrize("disable_uva", [False, True])
def test_cpu_offload(disable_pin_memory, disable_uva):
    env_vars = {
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": str(int(disable_pin_memory)),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": str(int(disable_uva)),
    }

    args = ["--cpu-offload-gb", "1"]

    # cuda graph only works with UVA offloading
    if disable_uva:
        args.append("--enforce-eager")

    compare_two_settings(
        model="hmellor/tiny-random-LlamaForCausalLM",
        arg1=[],
        arg2=args,
        env1=None,
        env2=env_vars,
    )


def test_cpu_offload_cuda_um_hints():
    try:
        from vllm.model_executor.offloader.cuda_memory_advice import (
            cuda_um_hints_supported,
        )
    except Exception as exc:
        pytest.skip(f"cuda_um_hints support probe unavailable: {exc}")

    if not cuda_um_hints_supported(0).supported:
        pytest.skip("requires supported CUDA full Unified Memory platform")

    compare_two_settings(
        model="hmellor/tiny-random-LlamaForCausalLM",
        arg1=[],
        arg2=[
            "--cpu-offload-gb",
            "1",
            "--offload-backend",
            "uva",
            "--offload-memory-advice",
            "cuda_um_hints",
        ],
        env1=None,
        env2=None,
    )
