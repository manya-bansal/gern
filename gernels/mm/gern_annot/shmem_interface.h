#pragma once

#include "../../current_path.h"
#include "annotations/shared_memory_manager.h"

namespace gern {
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
                  std::string(GERNELS_PATH) + "/mm/impl/sh_malloc.h",
              }) {
    }
};
}  // namespace gern