/* c/hip_debug.h — HIP debugging macros, included only when HIP_DEBUG=1.
 *
 * Usage: add `#include "hip_debug.h"` to backend_cuda.cu (after the
 * backend_gpu_compat.h include), then build with -DHIP_DEBUG to get
 * per-call HIP API error logging. Without HIP_DEBUG, HIP_CHECK is a no-op.
 *
 * Example:
 *   HIP_CHECK(hipMalloc(&ptr, size));
 *   HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
 */
#ifndef COLIBRI_HIP_DEBUG_H
#define COLIBRI_HIP_DEBUG_H

#ifdef HIP_DEBUG
#include <cstdio>
#include <cstdlib>

/* Logs the HIP error and aborts. Use for calls where failure is fatal. */
#define HIP_CHECK(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "[HIP] %s:%d: %s -> %s\n", \
                __FILE__, __LINE__, #call, hipGetErrorString(_err)); \
        std::abort(); \
    } \
} while(0)

/* Logs the HIP error but continues (for non-fatal warnings). */
#define HIP_CHECK_SOFT(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "[HIP] %s:%d: %s -> %s\n", \
                __FILE__, __LINE__, #call, hipGetErrorString(_err)); \
    } \
} while(0)

#else
#define HIP_CHECK(call) (call)
#define HIP_CHECK_SOFT(call) (call)
#endif

#endif