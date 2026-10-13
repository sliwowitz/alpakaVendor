/*
 * SPDX-FileCopyrightText: Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE(
    "A plan executed many times holds a bounded list of pending completions",
    "[unit][plan][pending]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, n);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, n);
        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(i), 0.0f};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
        constexpr std::size_t rounds = 50u, executions = 20u;
        for(std::size_t round = 0; round < rounds; ++round)
        {
            for(std::size_t i = 0; i < executions; ++i)
                alpaka::fft::onHost::executeForward(queue, plan, in, out);
            /* After the wait every execution so far is complete, so the next execution drops
             * them all: the list never holds more than the executions of one round. */
            alpaka::onHost::wait(queue);
            CHECK(plan.pendingCompletions() <= executions);
        }
        alpaka::fft::onHost::executeForward(queue, plan, in, out);
        CHECK(plan.pendingCompletions() == 1u);
        alpaka::onHost::wait(queue);
    }
}
