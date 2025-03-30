#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"
#include <iostream>
#include "annotations/lang_nodes.h"

namespace gern {
namespace annot {

class MatrixCPUTemplate : public AbstractDataType {
public:
	MatrixCPUTemplate(const std::string &name)
		: name(name) {
	}
	MatrixCPUTemplate()
		: MatrixCPUTemplate("test") {
	}
	std::string getName() const override {
		return name;
	}

	std::string getType() const override {
		return "gern::impl::MatrixCPUTemplate";
	}

	std::vector<Variable> getFields() const override {
		return {x, y, l_x, l_y};
	}

	FunctionSignature getAllocateFunction() const override {
		return FunctionSignature{
			.name = "gern::impl::MatrixCPUTemplate::allocate",
			.args = {x, y, l_x, l_y},
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
            .args = {x, y},
			.template_args = {l_x, l_y},
        };
	}
	FunctionSignature getQueryFunction() const override {
		return FunctionSignature{
			.name = "query",
            .args = {x, y},
			.template_args = {l_x, l_y},
		};
	}
	bool insertQuery() const override {
        return true;
    }

private:
	std::string name;
	Variable x{"x"};
	Variable y{"y"};
	Variable l_x{"l_x"};
	Variable l_y{"l_y"};
};

class MatrixCPUStatic : public MatrixCPUTemplate {
public:
	MatrixCPUStatic(const std::string &name, bool temp = false)
        : name(name), temp(temp) {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
		if (temp) {
			return "auto";
		}
		return "gern::impl::MatrixCPU";
    }

    std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::temp_matrix_allocate",
            .template_args = {l_x, l_y},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "dummy",
        };
    }
    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {x, y},
			.template_args = {l_x, l_y},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y},
			.template_args = {l_x, l_y},
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
	bool temp;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
};

class MatrixAddStaticStore : public AbstractFunction {
public:
    MatrixAddStaticStore()
        : input(new const MatrixCPUTemplate("input")),
          output(new const MatrixCPUTemplate("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix-template.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
		f.template_args = {l_x, l_y};
        return f;
    }

	

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
	Variable l_x{ "l_x" };
    Variable l_y{ "l_y" };
    Variable end{"end"};
};

// class MatrixDivn : public AbstractFunction {
// public:
//     MatrixDivn()
//         : input(new const MatrixCPU("input")),
//           output(new const MatrixCPU("output")) {
//     }
//     std::string getName() {
//         return "gern::impl::divn";
//     }

//     Annotation getAnnotation() override {

//         Variable x("x");
//         Variable y("y");
//         Variable l_x("l_x");
//         Variable l_y("l_y");

//         Variable row("row");
//         Variable col("col");

//         return annotate(For(x = Expr(0), output["row"], l_x,
//                             For(y = Expr(0), output["col"], l_y,
//                                 Produces::Subset(output, {x, y, l_x, l_y}),
//                                 Consumes::Subset(input, {x, y, l_x, l_y}))));
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::divn";
//         f.args = {Parameter(input), Parameter(n), Parameter(output)};
//         return f;
//     }

// protected:
//     AbstractDataTypePtr input;
// 	Variable n{"n", Datatype::Float32};
//     AbstractDataTypePtr output;
//     Variable end{"end"};
// };

// class MatrixSoftmax : public AbstractFunction {
// public:
//     MatrixSoftmax()
//         : input(new const MatrixCPU("input")),
//           output(new const MatrixCPU("output")) {
//     }
//     std::string getName() {
//         return "gern::impl::softmax";
//     }

//     Annotation getAnnotation() override {
//         Variable x("x");
// 		Variable y("y");
//         Variable l_x("l_x");
// 		Variable l_y("l_y");

//         return annotate(For(x = Expr(0), output["row"], l_x,
// 							For(y = Expr(0), output["col"], l_y,
// 								Produces::Subset(output, {x, y, l_x, l_y}),
// 								Consumes::Subset(input, {x, y, l_x, l_y}))));
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::softmax";
//         f.args = {Parameter(input), Parameter(output)};
//         return f;
//     }

// protected:
//     AbstractDataTypePtr input;
//     AbstractDataTypePtr output;
//     Variable end{"end"};
// };

// class MatrixTranspose : public AbstractFunction {
// public:
//     MatrixTranspose()
//         : input(new const MatrixCPU("input")),
//           output(new const MatrixCPU("output")) {
//     }
//     std::string getName() {
//         return "gern::impl::transpose";
//     }

//     Annotation getAnnotation() override {

//         Variable x("x");
//         Variable y("y");
//         Variable l_x("l_x");
//         Variable l_y("l_y");

//         Variable row("row");
//         Variable col("col");

//         return annotate(For(x = Expr(0), output["row"], l_x,
//                             For(y = Expr(0), output["col"], l_y,
//                                 Produces::Subset(output, {x, y, l_x, l_y}),
//                                 Consumes::Subset(input, {y, x, l_y, l_x}))));
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::transpose";
//         f.args = {Parameter(input), Parameter(output)};
//         return f;
//     }

// protected:
//     AbstractDataTypePtr input;
//     AbstractDataTypePtr output;
//     Variable end{"end"};
// };

// class MatrixMultiply : public AbstractFunction {
// public:
//     MatrixMultiply()
//         : a(new const MatrixCPU("inputA")),
// 		  b(new const MatrixCPU("inputB")),
//           output(new const MatrixCPU("output")) {
//     }
//     std::string getName() {
//         return "gern::impl::mmul";
//     }

//     Annotation getAnnotation() override {

//         Variable x("x");
//         Variable y("y");
//         Variable l_x("l_x");
//         Variable l_y("l_y");

//         Variable row("row");
//         Variable col("col");

// 		Variable shared_len("shared_len");

//         return annotate(For(x = Expr(0), output["row"], l_x,
//                             For(y = Expr(0), output["col"], l_y,
//                                 Produces::Subset(output, {x, y, l_x, l_y}),
//                                 Consumes::Subsets(
// 									SubsetObjMany({
// 										SubsetObj(a, {x, 0, l_x, shared_len}),
// 										SubsetObj(b, {0, y, shared_len, l_y})
// 									})))));
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::mmul";
//         f.args = {Parameter(a), Parameter(b), Parameter(output)};
//         return f;
//     }

// protected:
//     AbstractDataTypePtr a;
//     AbstractDataTypePtr b;
//     AbstractDataTypePtr output;
//     Variable end{"end"};
// };


// class SumRow : public AbstractFunction {
// public:
//     SumRow()
//         : input(new const MatrixCPU("input")),
//           output(new const ArrayCPU("output")) {
//     }

//     Annotation getAnnotation() override {

//         Variable x("x");
//         Variable y("y");
//         Variable l_x("l_x");
//         Variable l_y("l_y");

//         Variable row("row");
//         Variable col("col");

//         return annotate(For(x = Expr(0), output["size"], l_x,
//                             Produces::Subset(output, {x, l_x}),
//                             Consumes::Subset(input, {x, 0, l_x, col})));
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::sum_row";
//         f.args = {Parameter(input), Parameter(output)};
//         return f;
//     }

// protected:
//     AbstractDataTypePtr input;
//     AbstractDataTypePtr output;
//     Variable end{"end"};
// };

// class MaxRow : public SumRow {
// public:
//     MaxRow()
//         : SumRow() {
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::max_row";
//         f.args = {Parameter(input), Parameter(output)};
//         return f;
//     }
// };

// class ExpMatrix : public MatrixAddCPU {
// public:
//     ExpMatrix()
//         : MatrixAddCPU() {
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::exp_matrix";
//         f.args = {Parameter(input), Parameter(output)};
//         return f;
//     }
// };

// class SubtractVec : public AbstractFunction {
// public:
//     SubtractVec()
//         : input(new const MatrixCPU("input")),
//           output(new const MatrixCPU("output")),
//           vec(new const ArrayCPU("vec")) {
//     }

//     Annotation getAnnotation() override {

//         Variable x("x");
//         Variable y("y");
//         Variable l_x("l_x");
//         Variable l_y("l_y");

//         Variable row("row");
//         Variable col("col");

//         return annotate(For(x = Expr(0), output["row"], l_x,
//                             For(y = Expr(0), output["row"], l_y,
//                                 Produces::Subset(output, {x, y, l_x, l_y}),
//                                 Consumes::Subsets(
//                                     SubsetObjMany({
//                                         SubsetObj(input, {x, y, l_x, l_y}),
//                                         SubsetObj(vec, {x, l_x}),
//                                     })))));
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::subtract_vec";
//         f.args = {Parameter(vec), Parameter(input), Parameter(output)};
//         return f;
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

// protected:
//     AbstractDataTypePtr input;
//     AbstractDataTypePtr output;
//     AbstractDataTypePtr vec;
//     Variable end{"end"};
// };

// class DivideVec : public SubtractVec {
// public:
//     DivideVec()
//         : SubtractVec() {
//     }

//     virtual FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::divide_vec";
//         f.args = {Parameter(vec), Parameter(input), Parameter(output)};
//         return f;
//     }
// };

// class MatrixMultiplyCPU : public AbstractFunction {
// public:
//     MatrixMultiplyCPU()
//         : A(new const MatrixCPU("A")),
//           B(new const MatrixCPU("B")),
//           C(new const MatrixCPU("C")) {
//     }

//     Annotation getAnnotation() override {
//         Variable i("i");
//         Variable j("j");
//         Variable k("k");

//         Variable ti("ti", Datatype::Int64);
//         Variable tj("tj", Datatype::Int64);
//         Variable tk("tk", Datatype::Int64);

//         return annotate(For(i = Expr(0), ADTMember(C, "row", false), ti,
//                             For(j = Expr(0), ADTMember(C, "col", false), tj,
//                                 Produces::Subset(C, {i, j, ti, tj}),
//                                 Consumes::Subsets(
//                                     Reduce(k = Expr(0), ADTMember(A, "col", false), tk,
//                                            SubsetObjMany({
//                                                SubsetObj(A, {i, k, ti, tk}),
//                                                SubsetObj(B, {k, j, tk, tj}),
//                                            }))))));
//     }

//     FunctionSignature getFunction() override {
//         FunctionSignature f;
//         f.name = "gern::impl::matrix_multiply";
//         f.args = {Parameter(A), Parameter(B), Parameter(C)};
//         return f;
//     }

//     std::vector<std::string> getHeader() override {
//         return {
//             "cpu-matrix.h",
//         };
//     }

// private:
//     AbstractDataTypePtr A;
//     AbstractDataTypePtr B;
//     AbstractDataTypePtr C;
// };

}  // namespace annot
}  // namespace gern