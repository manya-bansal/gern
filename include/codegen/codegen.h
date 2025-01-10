#pragma once

#include "codegen/codegen_ir.h"
#include "compose/pipeline.h"
#include "compose/pipeline_visitor.h"
#include "utils/name_generator.h"

namespace gern {
namespace codegen {

class CodeGenerator : public PipelineVisitor {
public:
    CodeGenerator(std::string name = getUniqueName("function"),
                  std::string hook_prefix = "hook_")
        : name(name), hook_name(hook_prefix + name) {
    }

    CGStmt generate_code(const Pipeline &);

    using PipelineVisitor::visit;

    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);

    CGExpr gen(Expr, bool const_expr = false);  // Is this part of the LHS of a const_expr?
    CGExpr gen(Constraint);
    // Assign in used to track all the variables
    // that have been declared during lowering. The
    // const_expr tracks whether the assignment is
    // to a const_expr variable.
    CGStmt gen(Assign, bool const_expr);
    /**
     * @brief  Generate code expressions for Arguments.
     * This also tracks the input and output
     * data-structures for the pipeline,
     * and uses that information to expose
     * function arguments. The function takes an optional
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
    void insertInConstExpr(Variable);

    std::string getName() const;
    std::string getHookName() const;
    std::vector<std::string> getArgumentOrder() const;

private:
    // Little helper to make sure that
    // that once a var is declared, it's been
    // add to the declared set.
    CGExpr declVar(Variable v, bool const_expr);
    CGStmt setGrid(const IntervalNode *op);

    std::string name;
    std::string hook_name;
    std::set<Variable> declared;
    std::set<Variable> used;
    std::set<Variable> const_expr_vars;
    std::set<AbstractDataTypePtr> declared_adt;
    std::set<AbstractDataTypePtr> used_adt;
    CGStmt code;

    std::set<std::string> headers;
    std::set<std::string> includes;
    std::set<std::string> libs;
    std::vector<std::string> argument_order;

    // To track kernel launch parameters.
    struct cg_dim3 {
        CGExpr x = Literal::make(1);
        CGExpr y = Literal::make(1);
        CGExpr z = Literal::make(1);
        std::string str() {
            return x.str() + ", " +
                   y.str() + ", " +
                   z.str();
        }
    };

    cg_dim3 grid_dim;
    cg_dim3 block_dim;
};

}  // namespace codegen
}  // namespace gern