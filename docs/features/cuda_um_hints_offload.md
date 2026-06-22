# CUDA Unified Memory Hints for UVA Weight Offload

`cuda_um_hints` is an optional UVA weight-offload mode for supported CUDA
systems. It keeps selected offloaded weights in ordinary non-pinned CPU
system memory, applies `cudaMemAdviseSetReadMostly` and
`cudaMemAdviseSetAccessedBy`, and exposes those weights through
CUDA-visible system-memory tensor views.

## Usage

```bash
vllm serve "$MODEL" \
  --offload-backend uva \
  --cpu-offload-gb "$OFFLOAD_GB" \
  --offload-memory-advice cuda_um_hints
```

## Requirements

- Linux
- Not WSL
- CUDA 13.0+
- `--offload-backend uva`
- `--cpu-offload-gb > 0`
- Full Unified Memory support for pageable system memory
- Non-pinned CPU storage

## What Changes

- Advice is applied when weights are loaded and when offloaded weights are
  materialized again after post-load processing.
- The serving hot path does not add per-token `cudaMemAdvise` calls.
- Default UVA and prefetch offload behavior are unchanged when
  `--offload-memory-advice none`.

## Validation

- Compare the default UVA path, a system-unified CUDA view without advice,
  and the same view with `cuda_um_hints`.
- A microbenchmark is provided at
  `benchmarks/offload/benchmark_cuda_um_hints_offload.py`.

## Limits

- `--cpu-offload-gb` is still an offload budget for weights, not a memory
  guarantee.
- This mode does not change KV-cache allocation or scheduler behavior.
- Unsupported systems fail during startup when `cuda_um_hints` is
  requested.
