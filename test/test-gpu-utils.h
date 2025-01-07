namespace gern {
namespace test {

#include "compose/runner.h"
#include "config.h"

static gern::Runner::Options gpuRunner(std::string dir) {
    gern::Runner::Options o;
    o.filename = "test";
    o.prefix = "/tmp";
    o.include = " -I " + std::string(GERN_ROOT_DIR) +
                "/test/library/" + dir + "/impl";
    o.arch = std::string(GERN_CUDA_ARCH);
    return o;
}

}  // namespace test
}  // namespace gern