/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/sort/internal/api/config.hpp"

#if ALPAKAV_DEP_ONEDPL && ALPAKAV_HAS_ONEDPL

namespace alpaka::sort::internal
{
    /** oneDPL sorts a zip of the keys and the values with a stable sort on the queue's SYCL queue; the masked key
     * is the comparison. It wants no workspace of the caller's.
     */
    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    [[nodiscard]] std::size_t alpakaFnDispatch(
        WorkspaceBytesFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&&,
        PairsDescriptor<T_Key, T_Value> const&)
    {
        return 0u;
    }

    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SortPairsByKeyFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        using Unsigned = std::remove_cv_t<T_Key>;
        constexpr std::uint32_t width = sizeof(Unsigned) * 8u;
        Unsigned const low
            = pairs.endBit >= width ? ~Unsigned{0} : static_cast<Unsigned>((Unsigned{1} << pairs.endBit) - 1u);
        std::uint32_t const shift = pairs.beginBit;
        auto& syclQueue = queue.getNativeHandle();
        auto policy = oneapi::dpl::execution::make_device_policy(syclQueue);
        auto const n = static_cast<std::ptrdiff_t>(pairs.count);
        syclQueue.copy(pairs.keysIn, pairs.keysOut, pairs.count).wait();
        syclQueue.copy(pairs.valuesIn, pairs.valuesOut, pairs.count).wait();
        auto first = oneapi::dpl::make_zip_iterator(pairs.keysOut, pairs.valuesOut);
        oneapi::dpl::stable_sort(
            policy,
            first,
            first + n,
            [=](auto const& a, auto const& b)
            {
                return static_cast<Unsigned>((std::get<0>(a) & low) >> shift)
                       < static_cast<Unsigned>((std::get<0>(b) & low) >> shift);
            });
    }
} // namespace alpaka::sort::internal

#endif
