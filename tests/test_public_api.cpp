#include "ninfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ninfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ninfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ninfer::Engine>);
static_assert(!std::is_copy_constructible_v<ninfer::Engine>);

int main() {
    const ninfer::EngineOptions options;
    // Vision defaults off; the --vision-cpu field round-trips through EngineOptions.
    if (options.enable_vision || options.vision_cpu_offload) { return 1; }
    ninfer::EngineOptions on;
    on.enable_vision      = true;
    on.vision_cpu_offload = true;
    if (!on.enable_vision || !on.vision_cpu_offload) { return 2; }
    return 0;
}
