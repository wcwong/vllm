#include <cuda_runtime.h>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

namespace {

void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

}  // namespace

void cuda_advise_um_hints_for_tensor(torch::stable::Tensor& cpu_tensor,
                                     int64_t device_id) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(),
                  "CUDA Unified Memory hints tensor must be on CPU");

  STD_TORCH_CHECK(!is_pinned_cpu_tensor(cpu_tensor),
                  "CUDA Unified Memory hints require non-pinned CPU memory");

  STD_TORCH_CHECK(cpu_tensor.is_contiguous(),
                  "CUDA Unified Memory hints tensor must be contiguous");

  if (cpu_tensor.numel() == 0) {
    return;
  }

  void* ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
  size_t nbytes = static_cast<size_t>(cpu_tensor.numel()) *
                  static_cast<size_t>(cpu_tensor.element_size());

  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = static_cast<int>(device_id);

  check_cuda(
      cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetReadMostly, loc),
      "cudaMemAdviseSetReadMostly failed");

  check_cuda(
      cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, loc),
      "cudaMemAdviseSetAccessedBy failed");
}
