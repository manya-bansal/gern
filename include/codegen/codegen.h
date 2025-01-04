#ifndef GERN_CODEGEN_H
#define GERN_CODEGEN_H

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
    void visit(const Pipeline &);
    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);

    CGExpr genCodeExpr(Expr);
    CGExpr genCodeExpr(Constraint);
    // Assign in used to track all the variables
    // that have been declared during lowering.
    CGStmt genCodeExpr(Assign);
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
    CGExpr genCodeExpr(Argument a,
                       const std::map<AbstractDataTypePtr, AbstractDataTypePtr> &replacements = {});

    // To insert used variables.
    void insertInUsed(Variable);

    std::string getName() const;
    std::string getHookName() const;

private:
    std::string name;
    std::string hook_name;
    std::set<Variable> declared;
    std::set<Variable> used;
    std::set<AbstractDataTypePtr> declared_adt;
    std::set<AbstractDataTypePtr> used_adt;
    CGStmt code;

    std::set<std::string> headers;
    std::set<std::string> includes;
    std::set<std::string> libs;
};

}  // namespace codegen
}  // namespace gern

#endif