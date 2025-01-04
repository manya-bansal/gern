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
    CodeGenerator(std::string name = getUniqueName("function"))
        : name(name) {
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

    // To insert used variables.
    void insertInUsed(Variable);

    std::string getName() const;

private:
    std::string name;
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