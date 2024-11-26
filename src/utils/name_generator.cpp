#include "utils/name_generator.h"

namespace gern {

int counter = 0;

std::string getUniqueName(const std::string &prefix) {
  counter++;
  return prefix + "_" + std::to_string(counter);
}

} // namespace gern