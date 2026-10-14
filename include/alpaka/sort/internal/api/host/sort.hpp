/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "alpaka/sort/internal/api/config.hpp"

namespace alpaka::sort::internal
{
    /** The host sorts a permutation of the indices by the masked key with a stable sort, then gathers the keys and
     * the values through it. It wants no workspace: the permutation is its own, sized by the count.
     */
    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    [[nodiscard]] std::size_t alpakaFnDispatch(
        WorkspaceBytesFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&&,
        PairsDescriptor<T_Key, T_Value> const&)
    {
        return 0u;
    }

    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SortPairsByKeyFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        queue.enqueueNativeFn(
            [pairs](auto)
            {
                using Unsigned = std::remove_cv_t<T_Key>;
                constexpr std::uint32_t width = sizeof(Unsigned) * 8u;
                Unsigned const low
                    = pairs.endBit >= width ? ~Unsigned{0} : static_cast<Unsigned>((Unsigned{1} << pairs.endBit) - 1u);
                auto const masked = [=](Unsigned key) { return static_cast<Unsigned>((key & low) >> pairs.beginBit); };
                std::vector<std::size_t> order(pairs.count);
                std::iota(order.begin(), order.end(), std::size_t{0});
                std::stable_sort(
                    order.begin(),
                    order.end(),
                    [&](std::size_t a, std::size_t b) { return masked(pairs.keysIn[a]) < masked(pairs.keysIn[b]); });
                for(std::size_t i = 0u; i < pairs.count; ++i)
                {
                    pairs.keysOut[i] = pairs.keysIn[order[i]];
                    pairs.valuesOut[i] = pairs.valuesIn[order[i]];
                }
            });
    }
} // namespace alpaka::sort::internal
