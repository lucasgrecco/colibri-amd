/* backend_gpu_compat.h — one GPU backend source, two vendors.
 * Same pattern as compat.h for Windows: every platform difference lives in
 * this header and backend_cuda.cu stays untouched. Compiled by nvcc this is
 * a pass-through to the CUDA runtime (and mma.h for the tensor-core paths);
 * compiled by hipcc (ROCm, HIP=1) it maps the CUDA runtime surface
 * backend_cuda.cu uses onto HIP 1:1. The kernel language (__global__,
 * __shared__, <<<>>>) is shared syntax.
 *
 * COLI_GPU_HAS_WMMA: the WMMA tensor-core kernels are guarded by
 * __CUDA_ARCH__ >= 700 (device side) and by this flag at the host dispatch
 * sites. Under HIP, __CUDA_ARCH__ is not defined by hipcc, so we define it
 * here to 1100 (gfx1100 = RDNA3) to activate the WMMA kernel bodies.
 * rocWMMA provides float16_t and the fragment/mma_sync API; the CUDA
 * __half and nvcuda::wmma names are mapped to their rocWMMA equivalents.
 * Note: grouped_s4_wmma uses wmma::experimental::precision::s4 (4-bit)
 * which has no rocWMMA equivalent — that kernel stays a no-op under HIP
 * and falls back to quant_matmul. */
#ifndef COLIBRI_BACKEND_GPU_COMPAT_H
#define COLIBRI_BACKEND_GPU_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#define COLI_GPU_HAS_WMMA        1
#define __CUDA_ARCH__            700
#define __half                  rocwmma::float16_t
namespace nvcuda { namespace wmma = ::rocwmma; }
#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaGetErrorString       hipGetErrorString
#define cudaGetLastError         hipGetLastError
#define cudaSetDevice            hipSetDevice
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaDeviceProp           hipDeviceProp_t
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpy2D             hipMemcpy2D
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemGetInfo           hipMemGetInfo
#define cudaStream_t             hipStream_t
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking    hipStreamNonBlocking
#define cudaStreamDestroy        hipStreamDestroy
#define cudaStreamSynchronize    hipStreamSynchronize
#define cudaStreamWaitEvent      hipStreamWaitEvent
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaEvent_t              hipEvent_t
#define cudaEventCreate          hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming   hipEventDisableTiming
#define cudaEventDestroy         hipEventDestroy
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaMallocHost           hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyPeer           hipMemcpyPeer
#define cudaMemcpyPeerAsync      hipMemcpyPeerAsync
#define cudaMemsetAsync          hipMemsetAsync
#else
#include <cuda_runtime.h>
#include <mma.h>
#define COLI_GPU_HAS_WMMA        1
#endif

#endif