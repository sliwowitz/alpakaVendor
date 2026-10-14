/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "alpaka/sort/common.hpp"
#include "alpaka/sort/internal/api/config.hpp"

namespace alpaka::sort::internal
{
    /// The bytes of workspace a backend wants for sortPairsByKey over n pairs; the call takes no device time.
    ALPAKA_FN_SYMBOL(WorkspaceBytesFn);
    /// The sort of the pairs by their keys, enqueued on the queue.
    ALPAKA_FN_SYMBOL(SortPairsByKeyFn);

    /// The arguments every backend reads: the raw pointers and the count, checked once by the entry point.
    template<Key T_Key, Value T_Value>
    struct PairsDescriptor
    {
        T_Key const* keysIn;
        T_Value const* valuesIn;
        T_Key* keysOut;
        T_Value* valuesOut;
        std::size_t count;
        std::byte* workspace;
        std::size_t workspaceBytes;
        std::uint32_t beginBit;
        std::uint32_t endBit;
    };

    [[noreturn]] inline void refuse(std::string const& what)
    {
        throw std::invalid_argument("alpaka::sort: " + what);
    }
} // namespace alpaka::sort::internal

#include "alpaka/sort/internal/api/cuda/sort.hpp"
#include "alpaka/sort/internal/api/hip/sort.hpp"
#include "alpaka/sort/internal/api/host/sort.hpp"
#include "alpaka/sort/internal/api/oneapi/sort.hpp"
