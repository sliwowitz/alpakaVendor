/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../unit/test.hpp"
#include "alpaka/sort.hpp"

using namespace alpakaVendor::test;

namespace
{
    /// Keys below 2^bits from a hash of the index: many equal keys, in no order.
    std::uint32_t keyOf(std::size_t i, std::uint32_t bits)
    {
        std::uint64_t x = static_cast<std::uint64_t>(i) * 0x9E37'79B9'7F4A'7C15ull;
        x ^= x >> 29u;
        return static_cast<std::uint32_t>(x & ((1ull << bits) - 1u));
    }

    /** Sort n pairs whose keys hold a hash of the index in every bit, on the bits [beginBit, endBit) alone, and
     * compare with the standard library's stable sort on the same bits. The bits outside the range differ between
     * keys that the range calls equal, so a sort that read them would break the order of equal keys.
     */
    template<typename T_Key>
    void checkSortOnBits(auto& device, std::size_t n, std::uint32_t beginBit, std::uint32_t endBit)
    {
        auto queue = device.makeQueue();
        auto keysIn = alpaka::onHost::allocUnified<T_Key>(device, n);
        auto valuesIn = alpaka::onHost::allocUnified<std::uint32_t>(device, n);
        auto keysOut = alpaka::onHost::allocUnified<T_Key>(device, n);
        auto valuesOut = alpaka::onHost::allocUnified<std::uint32_t>(device, n);
        for(std::size_t i = 0u; i < n; ++i)
        {
            std::uint64_t x = (static_cast<std::uint64_t>(i) + 1u) * 0xD6E8'FEB8'6659'FD93ull;
            x ^= x >> 32u;
            keysIn[i] = static_cast<T_Key>(x * 0x9E37'79B9'7F4A'7C15ull);
            valuesIn[i] = static_cast<std::uint32_t>(i);
        }
        auto const options = alpaka::sort::Options{.beginBit = beginBit, .endBit = endBit};
        std::size_t const bytes
            = alpaka::sort::onHost::workspaceBytes(queue, keysIn, valuesIn, keysOut, valuesOut, options);
        auto workspace = alpaka::onHost::alloc<std::byte>(device, std::max<std::size_t>(bytes, 1u));
        alpaka::sort::onHost::sortPairsByKey(queue, keysIn, valuesIn, keysOut, valuesOut, workspace, options);
        alpaka::onHost::wait(queue);

        std::uint32_t const width = endBit - beginBit;
        T_Key const mask = width >= sizeof(T_Key) * 8u ? ~T_Key{0} : static_cast<T_Key>((T_Key{1} << width) - 1u);
        auto const bitsOf = [&](T_Key key) { return static_cast<T_Key>((key >> beginBit) & mask); };
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](std::size_t a, std::size_t b) { return bitsOf(keysIn[a]) < bitsOf(keysIn[b]); });
        bool keysMatch = true, valuesMatch = true;
        for(std::size_t i = 0u; i < n; ++i)
        {
            keysMatch = keysMatch && keysOut[i] == keysIn[order[i]];
            valuesMatch = valuesMatch && valuesOut[i] == static_cast<std::uint32_t>(order[i]);
        }
        CHECK(keysMatch);
        CHECK(valuesMatch);
    }
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "sortPairsByKey orders the pairs by key and keeps the order of equal keys",
    "[integr][sort]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isSortBackendEnabledForDevice(device))
    {
        SKIP("No sort backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr std::uint32_t n = 4099u;
        constexpr std::uint32_t bits = 7u;
        auto keysIn = alpaka::onHost::allocUnified<std::uint32_t>(device, n);
        auto valuesIn = alpaka::onHost::allocUnified<std::uint64_t>(device, n);
        auto keysOut = alpaka::onHost::allocUnified<std::uint32_t>(device, n);
        auto valuesOut = alpaka::onHost::allocUnified<std::uint64_t>(device, n);
        for(std::uint32_t i = 0u; i < n; ++i)
        {
            keysIn[i] = keyOf(i, bits) | (i % 3u == 0u ? (1u << 20u) : 0u);
            valuesIn[i] = i;
        }
        auto const options = alpaka::sort::Options{.beginBit = 0u, .endBit = bits};
        std::size_t const bytes
            = alpaka::sort::onHost::workspaceBytes(queue, keysIn, valuesIn, keysOut, valuesOut, options);
        auto workspace = alpaka::onHost::alloc<std::byte>(device, std::max<std::size_t>(bytes, 1u));

        alpaka::sort::onHost::sortPairsByKey(queue, keysIn, valuesIn, keysOut, valuesOut, workspace, options);
        alpaka::onHost::wait(queue);

        /* The reference: the same stable order from the standard library, on the low bits alone, so the bit above
         * them (set on every third key) does not take part. */
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](std::size_t a, std::size_t b) { return (keysIn[a] & 0x7Fu) < (keysIn[b] & 0x7Fu); });
        bool keysMatch = true, valuesMatch = true;
        for(std::uint32_t i = 0u; i < n; ++i)
        {
            keysMatch = keysMatch && keysOut[i] == keysIn[order[i]];
            valuesMatch = valuesMatch && valuesOut[i] == static_cast<std::uint64_t>(order[i]);
        }
        CHECK(keysMatch);
        CHECK(valuesMatch);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "sortPairsByKey keeps the order of equal keys over many chunks and several passes",
    "[integr][sort]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isSortBackendEnabledForDevice(device))
    {
        SKIP("No sort backend enabled for this alpaka API.");
    }
    else
    {
        constexpr std::size_t n = 1'000'003u;
        SECTION("three passes over bits [3, 20) of 32-bit keys")
        {
            checkSortOnBits<std::uint32_t>(device, n, 3u, 20u);
        }
        SECTION("two passes over bits [40, 56) of 64-bit keys")
        {
            checkSortOnBits<std::uint64_t>(device, n, 40u, 56u);
        }
        SECTION("one pass over the eight highest bits of 64-bit keys")
        {
            checkSortOnBits<std::uint64_t>(device, n, 56u, 64u);
        }
        SECTION("fewer pairs than one chunk, over every bit of 32-bit keys")
        {
            checkSortOnBits<std::uint32_t>(device, 1000u, 0u, 32u);
        }
    }
}
