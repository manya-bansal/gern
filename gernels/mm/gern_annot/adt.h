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
                   std::to_string(Col) + "," +
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

protected:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable row{"row", Datatype::Int64, true};
    Variable col{"col", Datatype::Int64, true};
    bool temp;
};

template<int Row, int Col, int Stride>
class MatrixGPUSequential : public MatrixGPU<Row, Col, Stride> {
public:
    MatrixGPUSequential(const std::string &name, bool temp)
        : MatrixGPU<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template query",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
    FunctionSignature getInsertFunction() const {
        return FunctionSignature{
            .name = "template insert",
            .args = {this->x, this->y},
        };
    }
};

class StaticArray : public AbstractDataType {
public:
    StaticArray(const std::string &name,
                bool temp = true)
        : name(name), temp(temp) {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        if (temp) {
            return "auto";
        }
        return "impl::StaticArray";
    }
    std::vector<Variable> getFields() const {
        return {x, len};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "impl::allocate_static_array",
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
            .name = "dummy",
        };
    }

    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "dummy",
        };
    }
    bool freeAlloc() const override {
        return false;
    }

private:
    std::string name;
    Variable x{"x"};
    Variable len{"size", Datatype::Int64, true};
    bool temp;
};

};  // namespace annot