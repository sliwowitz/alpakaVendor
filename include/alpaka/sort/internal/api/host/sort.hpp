/*
 * SPDX-FileCopyrightText: Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "alpaka/sort/internal/api/config.hpp"

namespace alpaka::sort::internal
{
    /** The host sorts by a least-significant-digit radix over the bits [beginBit, endBit) of the keys, eight bits
     * a pass. A pass is a stable counting sort of the pairs by one digit, in three launches on the queue: each
     * chunk of the pairs counts its digits, one worker turns the counts into the first place of each digit of each
     * chunk, and each chunk moves its pairs to their places in order. One worker walks a chunk from its first pair
     * to its last, and that keeps the order of equal keys. The launches take the executor of the host with the most
     * parallelism that the build has, and the queue orders them, so the sort enqueues and does not wait.
     */
    namespace hostRadix
    {
        /// The bits of one digit, and the digits a pass counts.
        inline constexpr std::uint32_t digitBits = 8u;
        inline constexpr std::size_t digits = std::size_t{1} << digitBits;

        /// The pairs of one chunk: enough to pay for a worker, and few enough that the chunks outnumber the workers.
        inline constexpr std::size_t chunkPairs = std::size_t{1} << 14u;
        /// The most chunks of a sort, which bounds the counts a pass keeps.
        inline constexpr std::size_t maxChunks = 1024u;

        [[nodiscard]] constexpr std::size_t chunksOf(std::size_t count) noexcept
        {
            return std::clamp<std::size_t>((count + chunkPairs - 1u) / chunkPairs, 1u, maxChunks);
        }

        [[nodiscard]] constexpr std::uint32_t passesOf(std::uint32_t beginBit, std::uint32_t endBit) noexcept
        {
            return (endBit - beginBit + digitBits - 1u) / digitBits;
        }

        /// The first pair of a chunk; the chunk ends where the next one begins.
        [[nodiscard]] constexpr std::size_t firstOf(std::size_t chunk, std::size_t count, std::size_t chunks) noexcept
        {
            return count / chunks * chunk + std::min(chunk, count % chunks);
        }

        [[nodiscard]] constexpr std::size_t aligned(std::size_t bytes) noexcept
        {
            constexpr std::size_t alignment = alignof(std::max_align_t);
            return (bytes + alignment - 1u) / alignment * alignment;
        }

        /** The workspace: the counts of every digit of every chunk, then, where a sort takes more than one pass, a
         * second buffer of keys and one of values that the passes alternate with the outputs.
         */
        template<Key T_Key, Value T_Value>
        struct Layout
        {
            std::size_t chunks;
            std::size_t countsBytes;
            std::size_t keysBytes;
            std::size_t valuesBytes;

            [[nodiscard]] static constexpr Layout of(PairsDescriptor<T_Key, T_Value> const& pairs) noexcept
            {
                std::size_t const chunks = chunksOf(pairs.count);
                bool const alternates = passesOf(pairs.beginBit, pairs.endBit) > 1u;
                return Layout{
                    chunks,
                    aligned(chunks * digits * sizeof(std::size_t)),
                    alternates ? aligned(pairs.count * sizeof(T_Key)) : 0u,
                    alternates ? aligned(pairs.count * sizeof(T_Value)) : 0u};
            }

            [[nodiscard]] constexpr std::size_t bytes() const noexcept
            {
                return countsBytes + keysBytes + valuesBytes;
            }
        };

        /// The digit of a key that one pass sorts by.
        template<Key T_Key>
        struct Digit
        {
            std::uint32_t shift;
            std::size_t mask;

            [[nodiscard]] ALPAKA_FN_ACC constexpr std::size_t operator()(T_Key key) const noexcept
            {
                return static_cast<std::size_t>(key >> shift) & mask;
            }
        };

        /// Each chunk counts the digits of its keys into its own row of the counts.
        struct CountKernel
        {
            template<Key T_Key>
            ALPAKA_FN_ACC void operator()(
                auto const& acc,
                T_Key const* keys,
                std::size_t count,
                std::size_t chunks,
                Digit<T_Key> digit,
                std::size_t* counts) const
            {
                for(auto chunk : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::size_t, 1u>{chunks}}))
                {
                    std::size_t* const row = counts + chunk.x() * digits;
                    std::fill_n(row, digits, std::size_t{0});
                    std::size_t const end = firstOf(chunk.x() + 1u, count, chunks);
                    for(std::size_t i = firstOf(chunk.x(), count, chunks); i < end; ++i)
                        ++row[digit(keys[i])];
                }
            }
        };

        /// One worker turns the counts into the first place of each digit of each chunk: the digits in order, and
        /// the chunks of one digit in order.
        struct PlaceKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, std::size_t chunks, std::size_t* counts) const
            {
                for(auto only : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::size_t, 1u>{1u}}))
                {
                    (void) only;
                    std::size_t running = 0u;
                    for(std::size_t d = 0u; d < digits; ++d)
                        for(std::size_t chunk = 0u; chunk < chunks; ++chunk)
                        {
                            std::size_t& cell = counts[chunk * digits + d];
                            std::size_t const here = cell;
                            cell = running;
                            running += here;
                        }
                }
            }
        };

        /// Each chunk moves its pairs, in order, to the places of their digits.
        struct MoveKernel
        {
            template<Key T_Key, Value T_Value>
            ALPAKA_FN_ACC void operator()(
                auto const& acc,
                T_Key const* keysFrom,
                T_Value const* valuesFrom,
                T_Key* keysTo,
                T_Value* valuesTo,
                std::size_t count,
                std::size_t chunks,
                Digit<T_Key> digit,
                std::size_t* places) const
            {
                for(auto chunk : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec<std::size_t, 1u>{chunks}}))
                {
                    std::size_t* const row = places + chunk.x() * digits;
                    std::size_t const end = firstOf(chunk.x() + 1u, count, chunks);
                    for(std::size_t i = firstOf(chunk.x(), count, chunks); i < end; ++i)
                    {
                        std::size_t const to = row[digit(keysFrom[i])]++;
                        keysTo[to] = keysFrom[i];
                        valuesTo[to] = valuesFrom[i];
                    }
                }
            }
        };
    } // namespace hostRadix

    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    [[nodiscard]] std::size_t alpakaFnDispatch(
        WorkspaceBytesFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&&,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        return hostRadix::Layout<T_Key, T_Value>::of(pairs).bytes();
    }

    template<Key T_Key, Value T_Value, alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SortPairsByKeyFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        PairsDescriptor<T_Key, T_Value> const& pairs)
    {
        using namespace hostRadix;
        using Idx = alpaka::Vec<std::size_t, 1u>;
        auto const layout = Layout<T_Key, T_Value>::of(pairs);
        auto* const counts = reinterpret_cast<std::size_t*>(pairs.workspace);
        auto* const keysSpare = reinterpret_cast<T_Key*>(pairs.workspace + layout.countsBytes);
        auto* const valuesSpare = reinterpret_cast<T_Value*>(pairs.workspace + layout.countsBytes + layout.keysBytes);
        auto const executor = alpaka::onHost::defaultExecutor(queue.getDevice());
        auto const chunked = alpaka::onHost::FrameSpec{Idx{layout.chunks}, Idx{1u}, executor};
        auto const alone = alpaka::onHost::FrameSpec{Idx{1u}, Idx{1u}, executor};

        /* The passes alternate between the outputs and the spare buffers, and the count of the passes decides which
         * one the first pass writes, so that the last pass writes the outputs. */
        std::uint32_t const passes = passesOf(pairs.beginBit, pairs.endBit);
        T_Key const* keysFrom = pairs.keysIn;
        T_Value const* valuesFrom = pairs.valuesIn;
        bool toOutputs = passes % 2u == 1u;
        for(std::uint32_t pass = 0u; pass < passes; ++pass)
        {
            std::uint32_t const shift = pairs.beginBit + pass * digitBits;
            std::uint32_t const width = std::min(digitBits, pairs.endBit - shift);
            Digit<T_Key> const digit{shift, (std::size_t{1} << width) - 1u};
            T_Key* const keysTo = toOutputs ? pairs.keysOut : keysSpare;
            T_Value* const valuesTo = toOutputs ? pairs.valuesOut : valuesSpare;
            queue.enqueue(chunked, alpaka::KernelBundle{CountKernel{}, keysFrom, pairs.count, layout.chunks, digit, counts});
            queue.enqueue(alone, alpaka::KernelBundle{PlaceKernel{}, layout.chunks, counts});
            queue.enqueue(
                chunked,
                alpaka::KernelBundle{
                    MoveKernel{},
                    keysFrom,
                    valuesFrom,
                    keysTo,
                    valuesTo,
                    pairs.count,
                    layout.chunks,
                    digit,
                    counts});
            keysFrom = keysTo;
            valuesFrom = valuesTo;
            toOutputs = !toOutputs;
        }
    }
} // namespace alpaka::sort::internal
