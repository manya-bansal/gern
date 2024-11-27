#ifndef GERN_TEST_UTILS_H
#define GERN_TEST_UTILS_H

#include "annotations/data_dependency_language.h"

namespace gern {
namespace dummy {

class TestDataStructure : public AbstractDataType {
public:
  std::string getName() const override { return "test"; }
};

} // namespace dummy
} // namespace gern

#endif