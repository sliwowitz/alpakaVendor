/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "alpaka/sort/common.hpp"
#include "alpaka/sort/internal/api/sort.hpp"

namespace alpaka::sort::onHost
{
    namespace internal
    {
        using namespace alpaka::sort::internal;

        template<typename T_View>
        [[nodiscard]] std::size_t countOf(T_View const& view)
        {
            return static_cast<std::size_t>(alpaka::onHost::getExtents(view)[0]);
        }

        /// Every view dense and of one count; the bits inside the key; the workspace as large as asked.
        template<typename T_KeysIn, typename T_ValuesIn, typename T_KeysOut, typename T_ValuesOut>
        [[nodiscard]] auto describe(
            T_KeysIn const& keysIn,
            T_ValuesIn const& valuesIn,
            T_KeysOut& keysOut,
            T_ValuesOut& valuesOut,
            std::byte* workspace,
            std::size_t workspaceBytes,
            Options const& options)
        {
            using KeyT = std::remove_cv_t<Value_t<T_KeysIn>>;
            using ValueT = std::remove_cv_t<Value_t<T_ValuesIn>>;
            static_assert(Key<KeyT>, "alpaka::sort orders unsigned integral keys");
            static_assert(Value<ValueT>, "alpaka::sort copies trivially copyable values");
            static_assert(
                std::same_as<KeyT, std::remove_cv_t<Value_t<T_KeysOut>>>,
                "the keys out take the type of the keys in");
            static_assert(
                std::same_as<ValueT, std::remove_cv_t<Value_t<T_ValuesOut>>>,
                "the values out take the type of the values in");
            static_assert(
                !std::is_const_v<Value_t<T_KeysOut>> && !std::is_const_v<Value_t<T_ValuesOut>>,
                "the outputs are written");
            std::size_t const count = countOf(keysIn);
            if(countOf(valuesIn) != count || countOf(keysOut) != count || countOf(valuesOut) != count)
                refuse("the four views must hold the same number of elements");
            for(auto const& pitch : {alpaka::onHost::getPitches(keysIn)[0], alpaka::onHost::getPitches(keysOut)[0]})
                if(static_cast<std::size_t>(pitch) != sizeof(KeyT))
                    refuse("the keys must be dense");
            for(auto const& pitch :
                {alpaka::onHost::getPitches(valuesIn)[0], alpaka::onHost::getPitches(valuesOut)[0]})
                if(static_cast<std::size_t>(pitch) != sizeof(ValueT))
                    refuse("the values must be dense");
            std::uint32_t const endBit = endBitOf<KeyT>(options);
            if(options.beginBit >= endBit || endBit > sizeof(KeyT) * 8u)
                refuse("the bits [beginBit, endBit) must lie inside the key and be a non-empty range");
            return PairsDescriptor<KeyT, ValueT>{
                alpaka::onHost::data(keysIn),
                alpaka::onHost::data(valuesIn),
                alpaka::onHost::data(keysOut),
                alpaka::onHost::data(valuesOut),
                count,
                workspace,
                workspaceBytes,
                options.beginBit,
                endBit};
        }
    } // namespace internal

    /** The bytes of workspace sortPairsByKey wants for these pairs on the device of the queue.
     *
     * The call takes no device time. A backend that keeps its own scratch answers zero.
     */
    [[nodiscard]] std::size_t workspaceBytes(
        auto& queue,
        concepts::VectorView auto const& keysIn,
        concepts::VectorView auto const& valuesIn,
        concepts::VectorView auto& keysOut,
        concepts::VectorView auto& valuesOut,
        Options options = {})
    {
        auto const pairs = internal::describe(keysIn, valuesIn, keysOut, valuesOut, nullptr, 0u, options);
        if(pairs.count == 0u)
            return 0u;
        return internal::WorkspaceBytesFn::call(queue, pairs);
    }

    /** Sort the pairs by their keys, ascending, and keep the order of pairs with equal keys.
     *
     * The outputs hold the result when the queue completes; the inputs are not written. The workspace is a view
     * of bytes at least as large as workspaceBytes() answered for these views, accessible to the device of the
     * queue; a backend that answered zero ignores it. The views must stay alive until the queue completes.
     * Only the bits [beginBit, endBit) of a key order the pairs.
     */
    void sortPairsByKey(
        auto& queue,
        concepts::VectorView auto const& keysIn,
        concepts::VectorView auto const& valuesIn,
        concepts::VectorView auto& keysOut,
        concepts::VectorView auto& valuesOut,
        concepts::VectorView auto& workspace,
        Options options = {})
    {
        static_assert(
            sizeof(std::remove_cv_t<internal::Value_t<decltype(workspace)>>) == 1u,
            "the workspace is a view of bytes");
        auto const pairs = internal::describe(
            keysIn,
            valuesIn,
            keysOut,
            valuesOut,
            reinterpret_cast<std::byte*>(alpaka::onHost::data(workspace)),
            internal::countOf(workspace),
            options);
        if(pairs.count == 0u)
            return;
        std::size_t const wanted = internal::WorkspaceBytesFn::call(queue, pairs);
        if(pairs.workspaceBytes < wanted)
            internal::refuse(
                "the workspace holds " + std::to_string(pairs.workspaceBytes) + " bytes, the sort wants "
                + std::to_string(wanted));
        internal::SortPairsByKeyFn::call(queue, pairs);
    }
} // namespace alpaka::sort::onHost
