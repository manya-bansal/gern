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
            .name = "...>",
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

template<int Row, int Col, int Stride>
class MatrixGlobalToGlobal : public MatrixGPU<Row, Col, Stride> {
public:
    MatrixGlobalToGlobal(const std::string &name, bool temp)
        : MatrixGPU<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template query_global_2_global",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
    FunctionSignature getInsertFunction() const {
        return FunctionSignature{
            .name = "template insert_global_2_global",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }

    bool insertQuery() const override {
        return false;  // Just a view, no insert.
    }
};

template<int Row, int Col, int Stride>
class MatrixQueryRegNoVector : public MatrixGPU<Row, Col, Stride> {
public:
    MatrixQueryRegNoVector(const std::string &name, bool temp)
        : MatrixGPU<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template query_2_reg_no_vector",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
    FunctionSignature getInsertFunction() const {
        return FunctionSignature{
            .name = "template insert_2_reg_no_vector",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }

    bool insertQuery() const override {
        return true;
    }
};

template<int Row, int Col, int Stride>
class MatrixGlobalToShared : public MatrixGPU<Row, Col, Stride> {
public:
    MatrixGlobalToShared(const std::string &name, bool temp)
        : MatrixGPU<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template stage_into_smem",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
    FunctionSignature getInsertFunction() const {
        return FunctionSignature{
            .name = "template insert_from_smem",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }

    FunctionSignature getFreeFunction() const {
        return FunctionSignature{
            .name = "free_smem",
        };
    }

    FunctionSignature getView() const {
        return FunctionSignature{
            .name = "template get_view",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }

    bool insertQuery() const override {
        return true;  // Just a view, no insert.
    }

    bool freeQuery() const override {
        return true;
    }

    bool freeAlloc() const override {
        return true;
    }
};

template<int Row, int Col, int Stride>
class MatrixGlobalToSharedVec : public MatrixGlobalToShared<Row, Col, Stride> {
public:
    MatrixGlobalToSharedVec(const std::string &name, bool temp)
        : MatrixGlobalToShared<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template stage_into_smem_vec",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
    FunctionSignature getViewVec() const {
        return FunctionSignature{
            .name = "template get_view_vec",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
        };
    }
};

template<int Row, int Col, int Stride>
class MatrixGlobalToSharedFlat : public MatrixGlobalToShared<Row, Col, Stride> {
public:
    MatrixGlobalToSharedFlat(const std::string &name, bool temp)
        : MatrixGlobalToShared<Row, Col, Stride>(name, temp) {
    }
    FunctionSignature getQueryFunction() const {
        return FunctionSignature{
            .name = "template stage_into_smem_flat",
            .args = {this->x, this->y},
            .template_args = {this->row, this->col},
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