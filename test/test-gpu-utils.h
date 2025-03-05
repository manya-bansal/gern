#include "compose/runner.h"
#include "config.h"
#include "test-utils.h"

namespace gern {
namespace test {

[[maybe_unused]] static gern::Runner::Options gpuRunner(const std::vector<std::string> &dirs) {
    gern::Runner::Options o = cpuRunner(dirs);
    o.filename = "test.cu";
    o.arch = std::string(GERN_CUDA_ARCH);
    return o;
}

[[maybe_unused]] static gern::Runner::Options gpuRunner(std::string dir) {
    return gpuRunner(std::vector<std::string>{dir});
}

class TrivialManager : public grid::SharedMemoryManager {
public:
    TrivialManager(Variable smem_size)
        : grid::SharedMemoryManager(
              FunctionCall{"init_shmem",
                           {smem_size},
                           {},
                           Parameter(),
                           LaunchArguments(),
                           LaunchArguments(),
                           DEVICE},
              {
                  std::string(GERN_ROOT_DIR) + "/smem_allocator/sh_malloc.cuh",
              }) {
    }
};

}  // namespace test
}  // namespace gern