#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

constexpr Invocation graph(std::int32_t t) {
    return {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true};
}

int q4_a16_conformance() {
    int failures = 0;

    constexpr std::array kN1024K5120{
        convenience(1), graph(2), a16(3),  graph(4),  a16(5),     a16(7),    graph(8), a16(9),
        a16(12),        a16(15),  a16(16), a16(17),   a16(24),    graph(25), a16(31),  a16(32),
        a16(33),        a16(48),  a16(55), graph(56), graph(57),  a16(58),   a16(63),  a16(64),
        a16(65),        a16(96),  a16(97), a16(127),  graph(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {1024, 5120, 101U, Comparison::Full, true, kN1024K5120});

    constexpr std::array kN1024K5120Bulk{
        a16(129),  a16(256),    a16(319),    graph(320), graph(321),  a16(322),
        a16(511),  graph(512),  a16(513),    a16(1023),  graph(1024), a16(1025),
        a16(1343), graph(1344), graph(1345), a16(1346),  a16(2048),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {1024, 5120, 101U, Comparison::Sampled, false, kN1024K5120Bulk});

    constexpr std::array kN4096K5120{
        a16(1), a16(2), a16(3), a16(4), a16(5), a16(6), a16(8), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {4096, 5120, 103U, Comparison::Sampled, false, kN4096K5120});

    // Full-output oracle covers each new mechanism and a masked capacity/column tile.
    constexpr std::array kN6144K5120Full{
        a16(1), graph(4), graph(8), graph(13), graph(24), graph(25), graph(64), graph(97),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {6144, 5120, 107U, Comparison::Full, true, kN6144K5120Full});

    constexpr std::array kN6144K5120{
        a16(2),     a16(3),     a16(7),    a16(9),      a16(15),    a16(16),  a16(17),  a16(23),
        a16(26),    a16(31),    a16(32),   a16(33),     a16(63),    a16(65),  a16(95),  a16(96),
        a16(98),    a16(127),   a16(128),  a16(129),    a16(191),   a16(192), a16(193), a16(383),
        a16(384),   graph(385), a16(386),  a16(511),    graph(512), a16(513), a16(639), a16(640),
        graph(641), a16(642),   a16(1023), graph(1024), a16(1025),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {6144, 5120, 107U, Comparison::Sampled, false, kN6144K5120});

    constexpr std::array kN5120K6144Full{
        convenience(1), graph(3),  graph(4),  graph(7),  graph(8),
        graph(13),      graph(16), graph(23), graph(24), graph(31),
        graph(32),      graph(33), graph(64), graph(97), graph(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {5120, 6144, 149U, Comparison::Full, true, kN5120K6144Full});

    constexpr std::array kN5120K6144{
        a16(2),    a16(5),      a16(9),     a16(15),   a16(17),  a16(25),    a16(34),
        a16(63),   a16(65),     a16(95),    a16(96),   a16(98),  a16(127),   a16(129),
        a16(191),  graph(192),  graph(193), a16(194),  a16(511), graph(512), a16(513),
        a16(1023), graph(1024), a16(1025),  a16(2048),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {5120, 6144, 149U, Comparison::Sampled, false, kN5120K6144});

    constexpr std::array kN7168K5120{
        a16(1),  a16(2),  a16(3),  a16(4),  a16(7),  a16(8),  a16(9),
        a16(10), a16(12), a16(15), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {7168, 5120, 109U, Comparison::Sampled, false, kN7168K5120});

    constexpr std::array kN34816K5120{
        a16(1), a16(2), a16(3), a16(4), a16(5), a16(6), a16(8), a16(16), a16(17), a16(18), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {34816, 5120, 113U, Comparison::Sampled, false, kN34816K5120});

    constexpr std::array kN131072K5120{
        a16(1), a16(2), a16(3), a16(4),   a16(5),   a16(6),
        a16(7), a16(8), a16(9), graph(3), graph(7), a16(128),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {131072, 5120, 127U, Comparison::Sampled, false, kN131072K5120});

    constexpr std::array kN131072K2048{
        a16(1),   a16(2),    a16(3),    a16(4),     a16(5),     a16(7),   a16(8),   a16(9),
        a16(12),  a16(15),   a16(16),   a16(17),    a16(19),    a16(20),  a16(21),  a16(32),
        a16(33),  a16(48),   a16(49),   a16(56),    a16(57),    a16(63),  a16(64),  a16(65),
        a16(72),  a16(73),   a16(80),   a16(81),    a16(96),    a16(97),  a16(103), a16(104),
        a16(105), a16(111),  a16(112),  a16(113),   a16(119),   a16(120), a16(121), a16(128),
        graph(3), graph(13), graph(19), graph(112), graph(120),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {131072, 2048, 131U, Comparison::Sampled, false, kN131072K2048});

    constexpr std::array kN3456K1152{
        a16(4),   a16(20),  a16(36),  a16(40),   a16(44),     a16(128),
        a16(320), a16(324), a16(328), a16(1024), a16(131072),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {3456, 1152, 137U, Comparison::Sampled, false, kN3456K1152});

    constexpr std::array kN4304K1152{
        a16(4),  a16(8),   a16(12),  a16(16),  a16(20),  a16(24),   a16(28),
        a16(32), a16(128), a16(320), a16(324), a16(328), a16(1024), a16(131072),
    };
    failures += run_shape("Q4_A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {4304, 1152, 139U, Comparison::Sampled, false, kN4304K1152});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q4_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
