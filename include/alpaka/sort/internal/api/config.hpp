/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_CUDA && __has_include(<cub/device/device_radix_sort.cuh>)
#    include <cub/device/device_radix_sort.cuh>
#    define ALPAKAV_HAS_CUB 1
#else
#    define ALPAKAV_HAS_CUB 0
#endif

#if ALPAKA_LANG_HIP && __has_include(<rocprim/rocprim.hpp>)
#    include <rocprim/rocprim.hpp>
#    define ALPAKAV_HAS_ROCPRIM 1
#else
#    define ALPAKAV_HAS_ROCPRIM 0
#endif

#if ALPAKAV_DEP_ONEDPL && ALPAKA_LANG_ONEAPI && __has_include(<oneapi/dpl/algorithm>)
#    include <oneapi/dpl/algorithm>
#    include <oneapi/dpl/execution>
#    include <oneapi/dpl/iterator>
#    define ALPAKAV_HAS_ONEDPL 1
#else
#    define ALPAKAV_HAS_ONEDPL 0
#endif
