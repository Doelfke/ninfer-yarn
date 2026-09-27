#pragma once

#include "ops/softmax_attention/dense/causal_cache/k8v4/geometry.h"

namespace ninfer::ops::detail {

template <int TokenTile, int Warps, int KeyTile, int MinBlocks = 1, bool DynamicArena = true>
struct K8V4KvGroupedMmaSchedule {
    static_assert(TokenTile > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static constexpr int kTokenTile     = TokenTile;
    static constexpr int kWarps         = Warps;
    static constexpr int kThreads       = Warps * 32;
    static constexpr int kKeyRows       = KeyTile;
    static constexpr int kMinBlocks     = MinBlocks;
    static constexpr bool kDynamicArena = DynamicArena;
    static constexpr int kArenaBytes    = 7 * KeyTile * 256 / 2;
};

// Two QK warps cover disjoint key halves; four PV warps split the output D axis.
template <int QueryTile = 64, int KeyTile = 64, int MaxRegisters = 120>
struct K8V4KvTiledMmaSchedule {
    static_assert(QueryTile == 16 || QueryTile == 32 || QueryTile == 64);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static_assert(MaxRegisters > 0 && MaxRegisters <= 255);
    static constexpr int kQueryRows       = QueryTile;
    static constexpr int kKeyRows         = KeyTile;
    static constexpr int kRowTiles        = QueryTile / 16;
    static constexpr int kDConsumers      = 4;
    static constexpr int kWarps           = kRowTiles * kDConsumers;
    static constexpr int kThreads         = kWarps * 32;
    static constexpr int kProducerWarps   = 2 * kRowTiles;
    static constexpr int kProducerThreads = kProducerWarps * 32;
    static constexpr int kMaxRegisters    = MaxRegisters;
    static constexpr int kQBytes          = QueryTile * 256;
    static constexpr int kQScaleBytes     = QueryTile * 4;
    static constexpr int kKBytes          = KeyTile * 256;
    static constexpr int kVBytes          = KeyTile * 128;
    static constexpr int kVStageBytes     = KeyTile * 256 * 2;
    static constexpr int kPBytes          = QueryTile * KeyTile * 2;
    static constexpr int kScaleBytes      = KeyTile * (2 + 16);
    static constexpr int kStatsBytes      = 7 * QueryTile * 4;
    static constexpr int kMainSharedBytes = kQBytes + kQScaleBytes + kKBytes + kVBytes +
                                            kVStageBytes + kPBytes + kScaleBytes + kStatsBytes;
    // The final FP32 inverse-rotation row tile aliases the completed mainloop arena.
    static constexpr int kSharedBytes =
        kMainSharedBytes > QueryTile * 256 * 4 ? kMainSharedBytes : QueryTile * 256 * 4;
    static_assert(kSharedBytes <= 99 * 1024);
};

// Inverse rotation consumes the complete normalized D256 row in one CTA.
struct K8V4KvMergeSchedule {
    static constexpr int kDChunk  = 256;
    static constexpr int kThreads = 256;
};

} // namespace ninfer::ops::detail
