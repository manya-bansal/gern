#pragma once

#include "annotations/abstract_function.h"
#include "annotations/data_dependency_language.h"
#include "compose/composable.h"

using namespace gern;

namespace annot {

class FloatPtr : public AbstractDataType {
public:
    FloatPtr(const std::string &name)
        : name(name) {
    }

    std::string getType() const override {
        return "float*";
    }

    std::string getName() const override {
        return name;
    }

    // No fields to return
    std::vector<Variable> getFields() const override {
        return {};
    }

    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{};
    }

    bool insertQuery() const override {
        return false;
    }

    bool freeQuery() const override {
        return false;
    }

private:
    std::string name;
};

class ArrayGPU : public AbstractDataType {
public:
    ArrayGPU(const std::string &name, bool temp = false)
        : name(name), temp(temp) {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        if (temp) {
            return "auto";
        }
        return "ArrayGPU";
    }

    std::vector<Variable> getFields() const override {
        return {x, len};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "allocate_local",
            .template_args = {len},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "dummy",
        };
    }

    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert_array",
            .args = {x},
            .template_args = {len},
        };
    }

    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x},
            .template_args = {len},
        };
    }

    bool freeAlloc() const override {
        return false;
    }

    bool insertQuery() const override {
        return true;
    }

private:
    std::string name;
    bool temp;
    Variable x{"x"};
    Variable len{"len"};
};

}  // namespace annot