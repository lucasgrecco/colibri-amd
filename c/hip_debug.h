/* c/hip_debug.h — HIP debugging macros, included only when HIP_DEBUG=1 */
#ifndef COLIBRI_HIP_DEBUG_H
#define COLIBRI_HIP_DEBUG_H

#ifdef HIP_DEBUG
#include <cstdio>
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::fprintf(stderr, "[HIP] %s:%d: %s → %s\n", \
                __FILE__, __LINE__, #call, hipGetErrorString(err)); \
    } \
} while(0)
#else
#define HIP_CHECK(call) call
#endif

#endif
