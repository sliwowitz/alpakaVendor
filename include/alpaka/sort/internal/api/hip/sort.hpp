/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/sort/internal/api/config.hpp"

#if ALPAKAV_HAS_ROCPRIM

#    include <hip/hip_runtime.h>
#    include <stdexcept>
#    include <string>

namespace alpaka::sort::internal
{
    inline void check(hipError_t status, char const* what)
    {
        if(status != hipSuccess)
            throw std::runtime_error(std::string("alpaka::sort: ") + what + ": " + hipGetErrorString(status));
    }

    /// rocPRIM's radix sort is stable and reads the bits [beginBit, endBit) of the key, as CUB's does.
    template<Key T_Key, Value T_Value>
    [[nodiscard]] std::size_t alpakaFnDispatch(
        WorkspaceBytesFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&&,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        std::size_t bytes = 0u;
        check(
            rocprim::radix_sort_pairs(
                nullptr,
                bytes,
                pairs.keysIn,
                pairs.keysOut,
                pairs.valuesIn,
                pairs.valuesOut,
                pairs.count,
                pairs.beginBit,
                pairs.endBit),
            "the workspace query of the radix sort");
        return bytes;
    }

    template<Key T_Key, Value T_Value>
    void alpakaFnDispatch(
        SortPairsByKeyFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        queue.enqueueNativeFn(
            [pairs](hipStream_t nativeStream)
            {
                std::size_t bytes = pairs.workspaceBytes;
                check(
                    rocprim::radix_sort_pairs(
                        pairs.workspace,
                        bytes,
                        pairs.keysIn,
                        pairs.keysOut,
                        pairs.valuesIn,
                        pairs.valuesOut,
                        pairs.count,
                        pairs.beginBit,
                        pairs.endBit,
                        nativeStream),
                    "the radix sort");
            });
    }
} // namespace alpaka::sort::internal

#endif
