/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/sort/internal/api/config.hpp"

#if ALPAKAV_HAS_CUB

#    include <cuda_runtime.h>
#    include <stdexcept>
#    include <string>

namespace alpaka::sort::internal
{
    inline void check(cudaError_t status, char const* what)
    {
        if(status != cudaSuccess)
            throw std::runtime_error(std::string("alpaka::sort: ") + what + ": " + cudaGetErrorString(status));
    }

    /** CUB's radix sort is stable and reads the bits [beginBit, endBit) of the key.
     *
     * The query with a null workspace returns the bytes CUB wants and touches no device.
     */
    template<Key T_Key, Value T_Value>
    [[nodiscard]] std::size_t alpakaFnDispatch(
        WorkspaceBytesFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&&,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        std::size_t bytes = 0u;
        check(
            cub::DeviceRadixSort::SortPairs(
                nullptr,
                bytes,
                pairs.keysIn,
                pairs.keysOut,
                pairs.valuesIn,
                pairs.valuesOut,
                pairs.count,
                static_cast<int>(pairs.beginBit),
                static_cast<int>(pairs.endBit)),
            "the workspace query of the radix sort");
        return bytes;
    }

    template<Key T_Key, Value T_Value>
    void alpakaFnDispatch(
        SortPairsByKeyFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        queue.enqueueNativeFn(
            [pairs](cudaStream_t nativeStream)
            {
                std::size_t bytes = pairs.workspaceBytes;
                check(
                    cub::DeviceRadixSort::SortPairs(
                        pairs.workspace,
                        bytes,
                        pairs.keysIn,
                        pairs.keysOut,
                        pairs.valuesIn,
                        pairs.valuesOut,
                        pairs.count,
                        static_cast<int>(pairs.beginBit),
                        static_cast<int>(pairs.endBit),
                        nativeStream),
                    "the radix sort");
            });
    }
} // namespace alpaka::sort::internal

#endif
