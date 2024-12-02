#ifndef GERN_TEST_UTILS_H
#define GERN_TEST_UTILS_H

#include "annotations/data_dependency_language.h"

namespace gern {
namespace dummy {

class TestDS : public AbstractDataType {
public:
    TestDS(const std::string &name)
        : name(name) {
    }
    TestDS()
        : TestDS("test") {
    }
    std::string getName() const override {
        return name;
    }

private:
    std::string name;
};

}  // namespace dummy

template<typename T>
static std::string getStrippedString(T e) {
    std::stringstream ss;
    ss << e;
    auto str = ss.str();
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return std::string(str);
}

}  // namespace gern

#endif