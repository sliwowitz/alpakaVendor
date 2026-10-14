/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "alpaka/sort.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE(
    "sort refuses views of different counts and bits outside the key before any dispatch",
    "[unit][sort][invalid]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    if constexpr(!isSortBackendEnabledForDevice(device))
    {
        SUCCEED();
    }
    else
    {
        auto queue = device.makeQueue();
        auto keys = alpaka::onHost::alloc<std::uint32_t>(device, 8u);
        auto values = alpaka::onHost::alloc<std::uint32_t>(device, 8u);
        auto keysOut = alpaka::onHost::alloc<std::uint32_t>(device, 8u);
        auto valuesOut = alpaka::onHost::alloc<std::uint32_t>(device, 8u);
        auto shortOut = alpaka::onHost::alloc<std::uint32_t>(device, 7u);
        auto workspace = alpaka::onHost::alloc<std::byte>(device, 1u);

        CHECK_THROWS_AS(
            alpaka::sort::onHost::sortPairsByKey(queue, keys, values, keysOut, shortOut, workspace),
            std::invalid_argument);
        CHECK_THROWS_AS(
            alpaka::sort::onHost::sortPairsByKey(
                queue,
                keys,
                values,
                keysOut,
                valuesOut,
                workspace,
                alpaka::sort::Options{.beginBit = 0u, .endBit = 33u}),
            std::invalid_argument);
        CHECK_THROWS_AS(
            alpaka::sort::onHost::sortPairsByKey(
                queue,
                keys,
                values,
                keysOut,
                valuesOut,
                workspace,
                alpaka::sort::Options{.beginBit = 5u, .endBit = 5u}),
            std::invalid_argument);
        CHECK_THROWS_AS(
            alpaka::sort::onHost::workspaceBytes(queue, keys, values, keysOut, shortOut),
            std::invalid_argument);
    }
}
