# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import TypeAdapter

import vllm.engine.arg_utils as arg_utils
from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig, UVAOffloadConfig
from vllm.engine.arg_utils import EngineArgs, FlexibleArgumentParser, get_kwargs


def test_cuda_um_hints_requires_cpu_offload_gb():
    with pytest.raises(ValueError, match="cpu_offload_gb > 0"):
        OffloadConfig(
            uva=UVAOffloadConfig(
                cpu_offload_gb=0,
                memory_advice="cuda_um_hints",
            )
        )


def test_cuda_um_hints_requires_uva_backend():
    with pytest.raises(ValueError, match="requires --offload-backend uva"):
        OffloadConfig(
            offload_backend="prefetch",
            uva=UVAOffloadConfig(
                cpu_offload_gb=1,
                memory_advice="cuda_um_hints",
            ),
        )


def test_cuda_um_hints_rejects_auto_prefetch():
    with pytest.raises(ValueError, match="use --offload-backend uva"):
        OffloadConfig(
            uva=UVAOffloadConfig(
                cpu_offload_gb=1,
                memory_advice="cuda_um_hints",
            ),
            prefetch=PrefetchOffloadConfig(offload_group_size=1),
        )


def test_cuda_um_hints_participates_in_hash():
    base = OffloadConfig(
        uva=UVAOffloadConfig(cpu_offload_gb=1),
    ).compute_hash()
    hinted = OffloadConfig(
        uva=UVAOffloadConfig(
            cpu_offload_gb=1,
            memory_advice="cuda_um_hints",
        ),
    ).compute_hash()

    assert base != hinted



def test_cuda_um_hints_dict_config_path_accepts_value():
    config = TypeAdapter(OffloadConfig).validate_python(
        {
            "uva": {
                "cpu_offload_gb": 1,
                "memory_advice": "cuda_um_hints",
            },
        }
    )

    assert config.uva.memory_advice == "cuda_um_hints"


def test_engine_args_parser_accepts_offload_memory_advice_values():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    hinted = parser.parse_args(["--offload-memory-advice", "cuda_um_hints"])
    default = parser.parse_args(["--offload-memory-advice", "none"])

    assert hinted.offload_memory_advice == "cuda_um_hints"
    assert default.offload_memory_advice == "none"


def test_engine_args_exposes_cuda_um_hints_choice_and_help(monkeypatch):
    monkeypatch.setattr(arg_utils, "NEEDS_HELP", True)
    arg_utils._compute_kwargs.cache_clear()
    kwargs = get_kwargs(EngineArgs)
    offload_memory_advice = kwargs["offload_memory_advice"]

    assert offload_memory_advice["choices"] == ["cuda_um_hints", "none"]
    assert (
        "CUDA managed-memory policy for UVA CPU weight offloading."
        in offload_memory_advice["help"]
    )
