#include <cuda_runtime.h>
#include <map>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <string>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#define CMM_CHECK(call)                                                        \
  do {                                                                         \
    std::string result = call;                                                 \
    if (result != "success") {                                                 \
      std::stringstream ss;                                                    \
      ss << "CMM call (" << #call << " ) failed with error: '" << result       \
         << "' (" __FILE__ << ":" << __LINE__ << ")\n";                        \
      throw sutil::Exception(ss.str().c_str());                                \
    }                                                                          \
  } while (0)

std::map<void *, size_t> memRecord;
std::string cmmMalloc(void **devPtr, size_t size) {
  if (memRecord.find(devPtr) != memRecord.end()) {
    return "cmmMalloc: devPtr already exists";
  }
  cudaError_t error = cudaMalloc(devPtr, size);
  if (error != cudaSuccess) {
    std::string cudaError = cudaGetErrorString(error);
    return "cuda Error:" + cudaError;
  }
  memRecord[*devPtr] = size;
  return "success";
}
std::string cmmMemcpy(void *dst, const void *src, size_t count,
                      cudaMemcpyKind kind) {
  // if (count > memRecord[dst]) {
  //     return "cmmMemset: count > memRecord[dst]";
  // }
  cudaError_t error = cudaMemcpy(dst, src, count, kind);
  if (error != cudaSuccess) {
    std::string cudaError = cudaGetErrorString(error);
    return "cuda Error:" + cudaError;
  }
  return "success";
}
std::string cmmFree(void *devPtr) {
  if (devPtr != 0 && memRecord.find(devPtr) == memRecord.end()) {
    return "cmmFree: devPtr not found";
  }
  cudaError_t error = cudaFree(devPtr);
  if (error != cudaSuccess) {
    std::string cudaError = cudaGetErrorString(error);
    return "cuda Error:" + cudaError;
  }
  if (devPtr != 0)
    memRecord.erase(devPtr);
  return "success";
}
std::string cmmMemset(void *devPtr, int value, size_t count) {
  // if (memRecord.find(devPtr) == memRecord.end()) {
  //     return "cmmFree: devPtr not found";
  // }
  // if (count > memRecord[devPtr]) {
  //     return "cmmMemset: count > memRecord[devPtr]";
  // }
  cudaError_t error = cudaMemset(devPtr, value, count);
  if (error != cudaSuccess) {
    std::string cudaError = cudaGetErrorString(error);
    return "cuda Error:" + cudaError;
  }
  return "success";
}
void cmmCheckMem() {
  int total = 0;
  for (auto it = memRecord.begin(); it != memRecord.end(); it++) {
    total += it->second;
  }
  if (total != 0)
    printf("GPU Memory leak! Now occupied cuda memory: %d\n", total);
}