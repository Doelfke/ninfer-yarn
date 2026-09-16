#pragma once
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {
enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SimtActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SimtBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SimtActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SimtBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SimtSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

} // namespace ninfer::ops::detail
