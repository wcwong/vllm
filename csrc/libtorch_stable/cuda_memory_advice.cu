#include <cuda_runtime.h>

#include <memory>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "ops.h"

namespace {

void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device_id) {
    check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice failed");
    if (previous_device_ != device_id) {
      check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
      restore_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (restore_) {
      cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_ = false;
};

void check_managed_pointer(void* ptr, int64_t device_id) {
  cudaPointerAttributes attrs{};
  check_cuda(cudaPointerGetAttributes(&attrs, ptr),
             "cudaPointerGetAttributes failed for managed allocation");

  STD_TORCH_CHECK(
      attrs.type == cudaMemoryTypeManaged,
      "expected cudaMallocManaged allocation, got cuda pointer type ",
      static_cast<int>(attrs.type));

  STD_TORCH_CHECK(
      attrs.device == static_cast<int>(device_id),
      "managed allocation is associated with unexpected CUDA device: ",
      attrs.device);

  STD_TORCH_CHECK(
      attrs.devicePointer != nullptr,
      "managed allocation is missing a device pointer");
  STD_TORCH_CHECK(attrs.hostPointer != nullptr,
                  "managed allocation is missing a host pointer");
}

}  // namespace

torch::stable::Tensor copy_to_managed_cuda_tensor(torch::stable::Tensor& src,
                                                  int64_t device_id) {
  STD_TORCH_CHECK(src.device().is_cpu() || src.device().is_cuda(),
                  "CUDA managed offload tensor must be on CPU or CUDA");
  STD_TORCH_CHECK(src.is_contiguous(),
                  "CUDA managed offload tensor must be contiguous");

  const int target_device = static_cast<int>(device_id);
  CudaDeviceGuard device_guard(target_device);

  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA,
                                       static_cast<int16_t>(target_device));

  if (src.numel() == 0) {
    return torch::stable::empty(src.sizes(), src.scalar_type(), src.layout(),
                                cuda_dev);
  }

  const size_t nbytes = static_cast<size_t>(src.numel()) *
                        static_cast<size_t>(src.element_size());

  void* ptr = nullptr;
  check_cuda(cudaMallocManaged(&ptr, nbytes), "cudaMallocManaged failed");

  std::unique_ptr<void, decltype(&cudaFree)> managed_ptr(ptr, &cudaFree);
  check_managed_pointer(ptr, target_device);

  check_cuda(cudaMemcpy(ptr, src.mutable_data_ptr(), nbytes, cudaMemcpyDefault),
             "copy into CUDA managed memory failed");

  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = target_device;

  check_cuda(
      cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetReadMostly, loc),
      "cudaMemAdviseSetReadMostly failed");

  check_cuda(
      cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, loc),
      "cudaMemAdviseSetAccessedBy failed");

  void* raw_ptr = managed_ptr.release();
  return torch::stable::from_blob(
      raw_ptr, src.sizes(), src.strides(), cuda_dev, src.scalar_type(),
      [](void* p) {
        cudaFree(p);
      });
}
