#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace annot {

class Scalar : public AbstractDataType {
public:
    Scalar(const std::string &name)
        : name(name) {
    }
    Scalar()
        : Scalar("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "float";
    }

    std::vector<Variable> getFields() const override {
        return {};
    }
    Function getAllocateFunction() const override {
        return Function{
            .name = "gern::impl::allocate_f32",
            .args = {},
        };
    }
    Function getFreeFunction() const override {
        return Function{
            .name = "destroy",
            .args = {},
        };
    }
    Function getInsertFunction() const override {
        return Function{
            .name = "insert",
            .args = {},
        };
    }
    Function getQueryFunction() const override {
        return Function{
            .name = "query",
            .args = {},
        };
    }

protected:
    std::string name;
};

}  // namespace annot
}  // namespace gern