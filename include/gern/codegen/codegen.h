#pragma once
#include <optional>

#include "annotations/std_less_specialization.h"
#include "codegen/codegen_ir.h"
#include "codegen/lower_visitor.h"
#include "utils/name_generator.h"
#include "utils/scoped_map.h"

#include <stack>

namespace gern {
namespace codegen {

template<typename T, typename Container = std::deque<T>>
class iterable_stack
    : public std::stack<T, Container> {
    using std::stack<T, Container>::c;

public:
    // expose just the iterators of the underlying container
    auto begin() {
        return std::begin(c);
    }
    auto end() {
        return std::end(c);
    }

    auto begin() const {
        return std::begin(c);
    }
    auto end() const {
        return std::end(c);
    }
};

class CodeGenerator : public LowerIRVisitor {
public:
    CodeGenerator(std::optional<std::vector<Parameter>> ordered_parameters = std::nullopt,
                  std::string name = getUniqueName("function"),
                  std::string hook_prefix = "hook_")
        : ordered_parameters(ordered_parameters), name(name), hook_name(hook_prefix + name) {
    }

    CGStmt generate_code(Composable);
    CGStmt top_level_codegen(LowerIR, bool is_device_launch);

    using LowerIRVisitor::visit;

    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    // void visit(const FunctionBoundary *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

    CGExpr gen(Expr);
    CGExpr gen(Constraint);
    CGExpr gen(AbstractDataTypePtr);
    CGStmt gen(FunctionCall f);
    CGStmt gen(MethodCall call);
    CGStmt gen(FunctionSignature f, CGStmt body);
    CGExpr gen(const Grid::Dim &p);
    Expr getExpr(const Grid::Dim &p) const;

    // Assign in used to track all the variables
    // that have been declared during lowering. The
    // const_expr tracks whether the assignment is
    // to a const_expr variable.
    CGStmt gen(Assign, bool const_expr = true);
    CGStmt gen(Expr a, Expr b);
    /**
     * @brief  Generate code expressions for Arguments.
     * This also tracks the input and output
     * data-structures for the pipeline,
     * and uses that information to expose
     * FunctionSignature arguments. The FunctionSignature takes an optional
     * replacement argument that substitutes data-structures
     * for an argument if provided.
     *
     * @param a The argument to lower.
     * @param replacements Optional map to make replacements.
     * @return CGExpr
     */
    CGExpr gen(Argument a);

    // To insert used variables.
    void insertInUsed(Variable);

    // Little helper to make sure that
    // that once a var is declared, it's been
    // add to the declared set.
    CGExpr declVar(Variable v, bool const_expr, bool track = true);
    CGExpr declADT(AbstractDataTypePtr, bool track = true, DeclProperties = DeclProperties());
    CGExpr declWithAuto(AbstractDataTypePtr ds, bool track);
    CGExpr declParameter(Parameter a,
                         bool track = true,
                         DeclProperties = DeclProperties());

    std::string getName() const;
    std::string getHookName() const;
    std::vector<std::string> getArgumentOrder() const;
    FunctionSignature getComputeFunctionSignature() const;

private:
    std::optional<std::vector<Parameter>> ordered_parameters;

    void updateGrid(Grid::Dim dim, Expr expr);

    CGStmt assertGrid(const Grid::Dim &dim);
    Expr getCurrentVal(const Grid::Dim &dim);

    std::vector<CGStmt> children;  // code generated for children.

    std::string name;
    std::string hook_name;
    std::set<Variable> declared;
    std::set<Variable> used;
    std::set<AbstractDataTypePtr> declared_adt;
    std::set<AbstractDataTypePtr> used_adt;
    // The signature of the generated function.
    FunctionSignature compute_func;
    CGStmt code;

    std::set<std::string> headers{"cassert"};  // add cassert, definitely using this.
    std::set<std::string> includes;
    std::set<std::string> libs;
    std::vector<std::string> argument_order;
    std::map<Grid::Dim, util::ScopedSet<Expr>> dims_defined;

    Variable smem_size;
};

}  // namespace codegen
}  // namespace gern