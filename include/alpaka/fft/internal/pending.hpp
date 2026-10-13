/*
 * SPDX-FileCopyrightText: Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace alpaka::fft::internal
{
    /** The executions of a plan that may still run on a queue.
     *
     * A plan records an event after each execution, so that its destructor can wait for the work that reads its
     * state. The list holds the executions that have not completed: before an event is recorded, every event that
     * is complete is dropped. A plan that lives as long as a run therefore holds a bounded list, however many times
     * it executes.
     */
    class PendingCompletions
    {
    public:
        PendingCompletions() = default;

        PendingCompletions(PendingCompletions&& other) noexcept
        {
            std::lock_guard lock(other.m_mutex);
            m_pending = std::move(other.m_pending);
        }

        PendingCompletions& operator=(PendingCompletions&& other) noexcept
        {
            if(this != &other)
            {
                std::scoped_lock lock(m_mutex, other.m_mutex);
                m_pending = std::move(other.m_pending);
            }
            return *this;
        }

        ~PendingCompletions()
        {
            wait();
        }

        /** Record the completion of the work enqueued so far, and drop the completions already reached. */
        void track(auto& queue)
        {
            auto event = queue.getDevice().makeEvent();
            queue.enqueue(event);
            std::lock_guard lock(m_mutex);
            std::erase_if(m_pending, [](Pending const& pending) { return pending.complete(); });
            m_pending.push_back(
                {[event] { return event.isComplete(); }, [event]() mutable { alpaka::onHost::wait(event); }});
        }

        /** Wait for every pending completion. */
        void wait() noexcept
        {
            std::vector<Pending> pending;
            {
                std::lock_guard lock(m_mutex);
                pending.swap(m_pending);
            }
            for(auto& entry : pending)
            {
                try
                {
                    entry.wait();
                }
                catch(...)
                {
                }
            }
        }

        /** How many executions have not completed. */
        [[nodiscard]] std::size_t size() const
        {
            std::lock_guard lock(m_mutex);
            return m_pending.size();
        }

    private:
        struct Pending
        {
            std::function<bool()> complete;
            std::function<void()> wait;
        };

        mutable std::mutex m_mutex;
        std::vector<Pending> m_pending;
    };
} // namespace alpaka::fft::internal
