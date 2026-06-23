# CUDA Unified Memory Hints for UVA Weight Offload

`cuda_um_hints` is an optional UVA weight-offload mode for supported CUDA
systems. It stores selected offloaded weights in CUDA managed memory via
`cudaMallocManaged`, copies the weight bytes into that managed allocation,
applies `cudaMemAdviseSetReadMostly` and `cudaMemAdviseSetAccessedBy`, and
exposes the result as a normal CUDA tensor.

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
- A managed-memory canary that proves `cudaMallocManaged`, tensor wrapping,
  `cudaMemAdviseSetReadMostly`, `cudaMemAdviseSetAccessedBy`, and GPU reads all
  work on the target platform

## What Changes

- Advice is applied when weights are loaded and when offloaded weights are
  materialized again after post-load processing.
- The serving hot path does not add per-token `cudaMemAdvise` calls.
- Default UVA and prefetch offload behavior are unchanged when
  `--offload-memory-advice none`.

## Validation

- Compare the default UVA path against the managed-memory copy path.
- A microbenchmark is provided at
  `benchmarks/offload/benchmark_cuda_um_hints_offload.py`.

## Limits

- `--cpu-offload-gb` is still an offload budget for weights, not a physical
  memory guarantee.
- Managed-memory pages may reside in system memory, HBM, or migrate under CUDA
  runtime and driver policy.
- This mode does not change KV-cache allocation or scheduler behavior.
- Unsupported systems fail during startup when `cuda_um_hints` is requested.
