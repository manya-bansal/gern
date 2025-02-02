#pragma once

#include "annotations/abstract_function.h"

using namespace gern;

namespace annot {

template<int Row, int Col, int Stride>
class MatrixGPU : public AbstractDataType {
public:
    MatrixGPU(const std::string &name, bool temp)
        : name(name), temp(temp) {
    }
    std::string getName() const override {
        return name;
    }
    std::string getType() const override {
        if (temp) {
            return "auto";
        } else {
            return "impl::MatrixGPU<" +
                   std::to_string(Row) + "," +
                   std::to_string(Col) + "," +
                   std::to_string(Row) + "," +
                   std::to_string(Stride) +
                   ">";
        }
    }

    std::vector<Variable> getFields() const {
        return {x, y, row, col};
    }
    FunctionSignature getAllocateFunction() const {
        return FunctionSignature{
            .name = "impl::allocate_static",
            .template_args = {row, col},
        };
    }
    FunctionSignature getFreeFunction() const {
        return FunctionSignature{
            .name = "destroy",
        };
    }
    FunctionSignature getInsertFunction() const {
        return FunctionSignature{
            .name = "template insert",
            .args = {x, y},
        };
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template query",
            .args = {x, y},
            .template_args = {row, col},
        };
    }
    bool insertQuery() const override {
        return true;
    }
    bool freeAlloc() const override {
        return false;
    }

private:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable row{"row", true};
    Variable col{"col", true};
    bool temp;
};

};  // namespace annot