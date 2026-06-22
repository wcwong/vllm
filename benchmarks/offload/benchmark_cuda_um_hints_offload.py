# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for CUDA Unified Memory hints on UVA weight offload.

Compares three paths:

- default UVA as configured by vLLM,
- a non-pinned system-memory CUDA view without advice, and
- a non-pinned system-memory CUDA view with cuda_um_hints.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from vllm.model_executor.offloader.cuda_memory_advice import (
    advise_cuda_um_hints_for_tensor,
    cuda_um_hints_supported,
    get_system_unified_cuda_view,
)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

DEFAULT_SIZES_GB = (1.0, 4.0, 16.0, 64.0)
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA Unified Memory hints for UVA offload."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--sizes-gb",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIZES_GB),
        help="Tensor sizes to benchmark, in GiB.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def _make_cpu_tensor(size_gb: float) -> torch.Tensor:
    bytes_per_elem = torch.empty((), dtype=torch.float16).element_size()
    num_bytes = max(1, int(size_gb * (1024**3)))
    numel = max(1, num_bytes // bytes_per_elem)
    return torch.full((numel,), 1.0, dtype=torch.float16, device="cpu")


def _assert_close(
    actual: float,
    expected: float,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> None:
    limit = max(atol, rtol * abs(expected))
    if math.isnan(actual) or math.isnan(expected) or abs(actual - expected) > limit:
        raise AssertionError(
            f"checksum mismatch: actual={actual!r} expected={expected!r} "
            f"(rtol={rtol}, atol={atol})"
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure_variant(
    *,
    name: str,
    view_fn,
    cpu_tensor: torch.Tensor,
    device: int,
    warmup: int,
    repeat: int,
    reference_checksum: float,
) -> dict[str, float | str | int]:
    for _ in range(warmup):
        view = view_fn(cpu_tensor, device)
        checksum = float(view.sum(dtype=torch.float64).item())
        _assert_close(checksum, reference_checksum)

    timings: list[float] = []
    checksum = reference_checksum
    for _ in range(repeat):
        start = time.perf_counter()
        view = view_fn(cpu_tensor, device)
        checksum = float(view.sum(dtype=torch.float64).item())
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        _assert_close(checksum, reference_checksum)

    avg_s = statistics.fmean(timings)
    bytes_read = cpu_tensor.numel() * cpu_tensor.element_size()

    return {
        "name": name,
        "avg_s": avg_s,
        "median_s": statistics.median(timings),
        "p95_s": _percentile(timings, 95.0),
        "gib_per_s": bytes_read / avg_s / (1024**3),
        "checksum": checksum,
    }


def _default_uva_view(cpu_tensor: torch.Tensor, device: int) -> torch.Tensor:
    del device
    return get_accelerator_view_from_cpu_tensor(cpu_tensor)


def _system_unified_view(cpu_tensor: torch.Tensor, device: int) -> torch.Tensor:
    return get_system_unified_cuda_view(cpu_tensor, device)


def _system_unified_view_with_hints(
    cpu_tensor: torch.Tensor,
    device: int,
) -> torch.Tensor:
    advise_cuda_um_hints_for_tensor(cpu_tensor, device)
    return get_system_unified_cuda_view(cpu_tensor, device)


def _main() -> int:
    args = _parse_args()

    support = cuda_um_hints_supported(args.device)
    if not support.supported:
        raise RuntimeError(
            "cuda_um_hints benchmark requires a supported CUDA full "
            f"Unified Memory platform: {support.reason}"
        )

    torch.cuda.set_device(args.device)

    results: dict[str, object] = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "runtime_version": support.runtime_version,
        "driver_version": support.driver_version,
        "attrs": support.attrs,
        "kernel_release": support.kernel_release,
        "sizes": [],
    }

    for size_gb in args.sizes_gb:
        cpu_tensor = _make_cpu_tensor(size_gb)
        reference_checksum = float(cpu_tensor.sum(dtype=torch.float64).item())
        size_result: dict[str, object] = {
            "size_gb": size_gb,
            "numel": cpu_tensor.numel(),
            "element_size": cpu_tensor.element_size(),
            "reference_checksum": reference_checksum,
            "variants": [],
        }

        variants = (
            ("default_uva", _default_uva_view),
            ("system_unified_view", _system_unified_view),
            ("system_unified_view_with_hints", _system_unified_view_with_hints),
        )
        for name, view_fn in variants:
            variant = _measure_variant(
                name=name,
                view_fn=view_fn,
                cpu_tensor=cpu_tensor,
                device=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                reference_checksum=reference_checksum,
            )
            size_result["variants"].append(variant)
            print(
                f"size={size_gb:g}GiB variant={name} "
                f"avg={variant['avg_s']:.6f}s "
                f"median={variant['median_s']:.6f}s "
                f"p95={variant['p95_s']:.6f}s "
                f"bw={variant['gib_per_s']:.2f}GiB/s"
            )

        results["sizes"].append(size_result)

    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
