#ifndef GERN_RANDOM_NAME_GENERATOR
#define GERN_RANDOM_NAME_GENERATOR

#include <string>

namespace gern {

std::string getUniqueName(const std::string &prefix = "gern_name");

} // namespace gern

#endif