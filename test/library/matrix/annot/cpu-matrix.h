#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"
#include <iostream>
#include "annotations/lang_nodes.h"

namespace gern {
namespace annot {

// class MatrixCPUNDim : public AbstractDataType {
// public:
//     MatrixCPUNDim(const std::string &name, int n)
//         : name(name), n(n) {
        
//     }
//     MatrixCPUNDim(int n)
//         : MatrixCPUNDim("test", n) {
        
//     }
//     std::string getName() const override {
//         return name;
//     }
//     std::string getType() const override {
//         return "gern::impl::MatrixCPUNDim";
//     }
// }

class MatrixCPU4Dim : public AbstractDataType {
public:
	MatrixCPU4Dim(const std::string &name)
		: name(name) {
	}
	MatrixCPU4Dim()
		: MatrixCPU4Dim("test") {
	}
	std::string getName() const override {
		return name;
	}

	std::string getType() const override {
		return "gern::impl::MatrixCPU4Dim";
	}

	std::vector<Variable> getFields() const override {
		return {w, x, y, z, l_w, l_x, l_y, l_z};
	}

	FunctionSignature getAllocateFunction() const override {
		return FunctionSignature{
			.name = "gern::impl::MatrixCPU4Dim::allocate",
			.args = {w, x, y, z, l_w, l_x, l_y, l_z},
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
			.args = {w, x, y, z, l_w, l_x, l_y, l_z},
		};
	}
	FunctionSignature getQueryFunction() const override {
		return FunctionSignature{
			.name = "query",
			.args = {w, x, y, z, l_w, l_x, l_y, l_z},
		};
	}

private:
	std::string name;
	Variable w{"w"};
	Variable x{"x"};
	Variable y{"y"};
	Variable z{"z"};
	Variable l_w{"l_w"};
	Variable l_x{"l_x"};
	Variable l_y{"l_y"};
	Variable l_z{"l_z"};
};
	

class MatrixCPU3Dim : public AbstractDataType {
public:
    MatrixCPU3Dim(const std::string &name)
        : name(name) {
    }
    MatrixCPU3Dim()
        : MatrixCPU3Dim("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::MatrixCPU3Dim";
    }

    std::vector<Variable> getFields() const override {
        return {x, y, z, l_x, l_y, l_z};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixCPU3Dim::allocate",
            .args = {x, y, z, l_x, l_y, l_z},
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
            .args = {x, y, z, l_x, l_y, l_z},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y, z, l_x, l_y, l_z},
        };
    }

private:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable z{"z"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
    Variable l_z{"l_z"};
};

class MatrixCPU : public AbstractDataType {
public:
    MatrixCPU(const std::string &name)
        : name(name) {
    }
    MatrixCPU()
        : MatrixCPU("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::MatrixCPU";
    }

    std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixCPU::allocate",
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
            .args = {x, y, l_x, l_y},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y, l_x, l_y},
        };
    }

private:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
};

class MatrixAddCPU3D : public AbstractFunction {
public:
	MatrixAddCPU3D()
		: input(new const MatrixCPU3Dim("input")),
			output(new const MatrixCPU3Dim("output")) {
	}
	std::string getName() {
		return "gern::impl::add";
	}

	Annotation getAnnotation() override {

		Variable x("x");
		Variable y("y");
		Variable z("z");
		Variable l_x("l_x");
		Variable l_y("l_y");
		Variable l_z("l_z");

		Variable row("row");
		Variable col("col");

		auto innerLoop = For(z = Expr(0), output["k_dim"], l_z, 
								Produces::Subset(output, {x, y, z, l_x, l_y, l_z}),
							Consumes::Subset(input, {x, y, z, l_x, l_y, l_z}));
		auto middleLoop = For(y = Expr(0), output["j_dim"], l_y, innerLoop);
		auto outerLoop = For(x = Expr(0), output["i_dim"], l_x, middleLoop);

		return annotate(outerLoop);
	}

	std::vector<std::string> getHeader() override {
		return {
			"cpu-matrix.h",
		};
	}

	virtual FunctionSignature getFunction() override {
		FunctionSignature f;
		f.name = "gern::impl::add";
		f.args = {Parameter(input), Parameter(output)};
		return f;
	}

protected:
	AbstractDataTypePtr input;
	AbstractDataTypePtr output;
	Variable end{"end"};
};

class MatrixAddCPU : public AbstractFunction {
public:
    MatrixAddCPU()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixAttention4D : public AbstractFunction {
public:
    MatrixAttention4D()
        : q(new const MatrixCPU4Dim("q")),
            k(new const MatrixCPU4Dim("k")),
            v(new const MatrixCPU4Dim("v")),
            output(new const MatrixCPU4Dim("output")) {
    }
    std::string getName() {
        return "gern::impl::attention";
    }

    Annotation getAnnotation() override {

        Variable w("w");
        Variable x("x");
        Variable y("y");
        Variable z("z");
        Variable l_w("l_w");
        Variable l_x("l_x");
        Variable l_y("l_y");
        Variable l_z("l_z");
        Variable height("height");
        Variable width("width");

        auto twoDimLoop = For(y = Expr(0), output["k_dim"], l_y,
                                For(z = Expr(0), output["l_dim"], l_z,
                                    Produces::Subset(output, {w, x, y, z, l_w, l_x, l_y, l_z}),
                                    Consumes::Subsets(
                                        SubsetObjMany({
                                            SubsetObj(q, {w, x, y, 0, l_w, l_x, l_y, width}),
                                            SubsetObj(k, {w, x, 0, 0, l_w, l_x, height, width}),
                                            SubsetObj(v, {w, x, 0, z, l_w, l_x, height, l_z}),
                                        }))));
        
        auto outerLoop = For(w = Expr(0), output["i_dim"], l_w,
                            For(x = Expr(0), output["j_dim"], l_x, twoDimLoop));

        return annotate(outerLoop);
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::attention";
        f.args = {Parameter(q), Parameter(k), Parameter(v), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr q;
    AbstractDataTypePtr k;
    AbstractDataTypePtr v;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixAttention : public AbstractFunction {
public:
    MatrixAttention()
        : q(new const MatrixCPU("q")),
            k(new const MatrixCPU("k")),
            v(new const MatrixCPU("v")),
            output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::attention";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        Variable height("height");
        Variable width("width");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subsets(
                                    SubsetObjMany({
                                        SubsetObj(q, {x, 0, l_x, width}),
                                        SubsetObj(k, {0, 0, height, width}),
                                        SubsetObj(v, {0, y, height, l_y}),
                                    })))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::attention";
        f.args = {Parameter(q), Parameter(k), Parameter(v), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr q;
    AbstractDataTypePtr k;
    AbstractDataTypePtr v;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixDivn : public AbstractFunction {
public:
    MatrixDivn()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::divn";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::divn";
        f.args = {Parameter(input), Parameter(n), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    Variable n{"n", Datatype::Float32};
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixDivn4D : public AbstractFunction {
public:
	MatrixDivn4D()
		: input(new const MatrixCPU4Dim("input")),
			output(new const MatrixCPU4Dim("output")) {
	}
	std::string getName() {
		return "gern::impl::divn";
	}

	Annotation getAnnotation() override {

		Variable w("w");
		Variable x("x");
		Variable y("y");
		Variable z("z");
		Variable l_w("l_w");
		Variable l_x("l_x");
		Variable l_y("l_y");
		Variable l_z("l_z");

		auto innerLoop = For(z = Expr(0), output["l_dim"], l_z, 
								Produces::Subset(output, {w, x, y, z, l_w, l_x, l_y, l_z}),
							Consumes::Subset(input, {w, x, y, z, l_w, l_x, l_y, l_z}));
		auto middleLoop = For(y = Expr(0), output["k_dim"], l_y, innerLoop);
		auto secondMiddleLoop = For(x = Expr(0), output["j_dim"], l_x, middleLoop);
		auto outerLoop = For(w = Expr(0), output["i_dim"], l_w, secondMiddleLoop);

		return annotate(outerLoop);
	}

	std::vector<std::string> getHeader() override {
		return {
			"cpu-matrix.h",
		};
	}

	virtual FunctionSignature getFunction() override {
		FunctionSignature f;
		f.name = "gern::impl::divn";
		f.args = {Parameter(input), Parameter(n), Parameter(output)};
		return f;
	}

protected:
	AbstractDataTypePtr input;
	Variable n{"n", Datatype::Float32};
	AbstractDataTypePtr output;
	Variable end{"end"};
};

class MatrixSoftmax4D : public AbstractFunction {
public:
    MatrixSoftmax4D()
        : input(new const MatrixCPU4Dim("input")),
            output(new const MatrixCPU4Dim("output")) {
    }
    std::string getName() {
        return "gern::impl::softmax";
    }

    Annotation getAnnotation() override {
        Variable w("w");
        Variable x("x");
        Variable y("y");
        Variable z("z");
        Variable l_w("l_w");
        Variable l_x("l_x");
        Variable l_y("l_y");
        Variable l_z("l_z");

        auto innerLoop = For(y = Expr(0), output["k_dim"], l_y,
                            For(z = Expr(0), output["l_dim"], l_z,
                                Produces::Subset(output, {w, x, y, z, l_w, l_x, l_y, l_z}),
                                Consumes::Subset(input, {w, x, y, z, l_w, l_x, l_y, l_z})));

        auto outerLoop = For(w = Expr(0), output["i_dim"], l_w, 
                            For(x = Expr(0), output["j_dim"], l_x, innerLoop));
        return annotate(outerLoop);
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::softmax";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixSoftmax : public AbstractFunction {
public:
    MatrixSoftmax()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::softmax";
    }

    Annotation getAnnotation() override {
        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::softmax";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixTranspose4D : public AbstractFunction {
public:
    MatrixTranspose4D()
        : input(new const MatrixCPU4Dim("input")),
            output(new const MatrixCPU4Dim("output")) {
    }
    std::string getName() {
        return "gern::impl::transpose2d";
    }

    Annotation getAnnotation() override {

        Variable w("w");
        Variable x("x");
        Variable y("y");
        Variable z("z");
        Variable l_w("l_w");
        Variable l_x("l_x");
        Variable l_y("l_y");
        Variable l_z("l_z");

        Variable row("row");
        Variable col("col");

        auto innerLoop = For(y = Expr(0), output["k_dim"], l_y,
                            For(z = Expr(0), output["l_dim"], l_z,
                                Produces::Subset(output, {w, x, y, z, l_w, l_x, l_y, l_z}),
                                Consumes::Subset(input, {w, x, z, y, l_w, l_x, l_z, l_y})));
        auto outerLoop = For(w = Expr(0), output["i_dim"], l_w,
                            For(x = Expr(0), output["j_dim"], l_x, innerLoop));

        return annotate(outerLoop);
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::transpose2d";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixTranspose : public AbstractFunction {
public:
    MatrixTranspose()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::transpose";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {y, x, l_y, l_x}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::transpose";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixMultiply4D : public AbstractFunction {
public:
    MatrixMultiply4D()
        : a(new const MatrixCPU4Dim("inputA")),
            b(new const MatrixCPU4Dim("inputB")),
            output(new const MatrixCPU4Dim("output")) {
    }
    std::string getName() {
        return "gern::impl::mmul2d";
    }

    Annotation getAnnotation() override {

        Variable w("w");
        Variable x("x");
        Variable y("y");
        Variable z("z");
        Variable l_w("l_w");
        Variable l_x("l_x");
        Variable l_y("l_y");
        Variable l_z("l_z");

        Variable row("row");
        Variable col("col");

        Variable shared_len("shared_len");

        auto innerLoop = For(y = Expr(0), output["k_dim"], l_y,
                            For(z = Expr(0), output["l_dim"], l_z,
                                Produces::Subset(output, {w, x, y, z, l_w, l_x, l_y, l_z}),
                                Consumes::Subsets(
                                    SubsetObjMany({
                                        SubsetObj(a, {w, x, y, 0, l_w, l_x, l_y, shared_len}),
                                        SubsetObj(b, {w, x, 0, z, l_w, l_x, shared_len, l_z})
                                    }))));
        
        auto outerLoop = For(w = Expr(0), output["i_dim"], l_w, 
                            For(x = Expr(0), output["j_dim"], l_x, innerLoop));

        return annotate(outerLoop);
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::mmul2d";
        f.args = {Parameter(a), Parameter(b), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr a;
    AbstractDataTypePtr b;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixMultiply : public AbstractFunction {
public:
    MatrixMultiply()
        : a(new const MatrixCPU("inputA")),
          b(new const MatrixCPU("inputB")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::mmul";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        Variable shared_len("shared_len");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subsets(
                                    SubsetObjMany({
                                        SubsetObj(a, {x, 0, l_x, shared_len}),
                                        SubsetObj(b, {0, y, shared_len, l_y})
                                    })))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::mmul";
        f.args = {Parameter(a), Parameter(b), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr a;
    AbstractDataTypePtr b;
    AbstractDataTypePtr output;
    Variable end{"end"};
};


class SumRow : public AbstractFunction {
public:
    SumRow()
        : input(new const MatrixCPU("input")),
          output(new const ArrayCPU("output")) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["size"], l_x,
                            Produces::Subset(output, {x, l_x}),
                            Consumes::Subset(input, {x, 0, l_x, col})));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::sum_row";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MaxRow : public SumRow {
public:
    MaxRow()
        : SumRow() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::max_row";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }
};

class ExpMatrix : public MatrixAddCPU {
public:
    ExpMatrix()
        : MatrixAddCPU() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::exp_matrix";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }
};

class SubtractVec : public AbstractFunction {
public:
    SubtractVec()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")),
          vec(new const ArrayCPU("vec")) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["row"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subsets(
                                    SubsetObjMany({
                                        SubsetObj(input, {x, y, l_x, l_y}),
                                        SubsetObj(vec, {x, l_x}),
                                    })))));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::subtract_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    AbstractDataTypePtr vec;
    Variable end{"end"};
};

class DivideVec : public SubtractVec {
public:
    DivideVec()
        : SubtractVec() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::divide_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }
};

class MatrixMultiplyCPU : public AbstractFunction {
public:
    MatrixMultiplyCPU()
        : A(new const MatrixCPU("A")),
          B(new const MatrixCPU("B")),
          C(new const MatrixCPU("C")) {
    }

    Annotation getAnnotation() override {
        Variable i("i");
        Variable j("j");
        Variable k("k");

        Variable ti("ti", Datatype::Int64);
        Variable tj("tj", Datatype::Int64);
        Variable tk("tk", Datatype::Int64);

        return annotate(For(i = Expr(0), ADTMember(C, "row", false), ti,
                            For(j = Expr(0), ADTMember(C, "col", false), tj,
                                Produces::Subset(C, {i, j, ti, tj}),
                                Consumes::Subsets(
                                    Reduce(k = Expr(0), k_dim, tk,
                                           SubsetObjMany({
                                               SubsetObj(A, {i, k, ti, tk}),
                                               SubsetObj(B, {k, j, tk, tj}),
                                           }))))));
    }

    FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::matrix_multiply";
        f.args = {Parameter(A), Parameter(B), Parameter(C), k_dim};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

private:
    AbstractDataTypePtr A;
    AbstractDataTypePtr B;
    AbstractDataTypePtr C;
    Variable k_dim{"k_dim", Datatype::Int64};
};

}  // namespace annot
}  // namespace gern