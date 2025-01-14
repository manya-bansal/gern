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
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::allocate_f32",
            .args = {},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
            .args = {},
        };
    }
    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {},
        };
    }

protected:
    std::string name;
};

}  // namespace annot
}  // namespace gern