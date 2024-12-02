#include "utils/name_generator.h"

#include <atomic>

namespace gern {

std::atomic<int> counter = 0;

std::string getUniqueName(const std::string &prefix) {
    counter++;
    return prefix + "_" + std::to_string(counter);
}

}  // namespace gern