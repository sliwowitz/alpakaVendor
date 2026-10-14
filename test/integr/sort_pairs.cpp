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
