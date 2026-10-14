/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace alpaka::sort
{
    /** The bits of a key that the sort reads: [beginBit, endBit).
     *
     * A key known to lie below 2^k needs endBit = k, and a radix sort then makes fewer passes. An endBit of zero
     * means every bit of the key type.
     */
    struct Options
    {
        std::uint32_t beginBit = 0u;
        std::uint32_t endBit = 0u;
    };

    /// A key the sort orders: an unsigned integer, so that the radix of every backend reads it the same.
    template<typename T>
    concept Key = std::unsigned_integral<std::remove_cv_t<T>>;

    /// A value that travels with its key: copied, never constructed, by every backend.
    template<typename T>
    concept Value = std::is_trivially_copyable_v<std::remove_cv_t<T>>;

    namespace concepts
    {
        /// A one-dimensional alpaka view.
        template<typename T>
        concept VectorView = alpaka::concepts::IMdSpan<std::remove_cvref_t<T>> && std::remove_cvref_t<T>::dim() == 1u;
    } // namespace concepts

    namespace internal
    {
        template<typename T_View>
        using Value_t = alpaka::GetValueType_t<std::remove_cvref_t<T_View>>;

        /// The bits the sort reads of a key of type T under the options: the whole key where endBit is zero.
        template<Key T>
        [[nodiscard]] constexpr std::uint32_t endBitOf(Options const& options) noexcept
        {
            return options.endBit == 0u ? static_cast<std::uint32_t>(sizeof(T) * 8u) : options.endBit;
        }
    } // namespace internal
} // namespace alpaka::sort
