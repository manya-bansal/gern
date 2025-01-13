#include "compose/runner.h"
#include "config.h"
#include "test-utils.h"

namespace gern {
namespace test {

[[maybe_unused]] static gern::Runner::Options gpuRunner(const std::vector<std::string> &dirs) {
    gern::Runner::Options o = cpuRunner(dirs);
    o.arch = std::string(GERN_CUDA_ARCH);
    return o;
}

[[maybe_unused]] static gern::Runner::Options gpuRunner(std::string dir) {
    return gpuRunner(std::vector<std::string>{dir});
}

}  // namespace test
}  // namespace gern