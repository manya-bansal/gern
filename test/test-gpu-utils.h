#include "compose/runner.h"
#include "config.h"
#include "test-utils.h"

namespace gern {
namespace test {

static gern::Runner::Options gpuRunner(std::string dir) {
    gern::Runner::Options o = cpuRunner(dir);
    o.arch = std::string(GERN_CUDA_ARCH);
    return o;
}

}  // namespace test
}  // namespace gern