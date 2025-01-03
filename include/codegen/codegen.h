#ifndef GERN_CODEGEN_H
#define GERN_CODEGEN_H

#include "codegen/codegen_ir.h"
#include "compose/pipeline.h"
#include "compose/pipeline_visitor.h"

namespace gern {
namespace codegen {

class CodeGenerator : public PipelineVisitor {
public:
    CodeGenerator() = default;
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

    static CGExpr genCodeExpr(Expr);
    static CGExpr genCodeExpr(Constraint);
    static CGStmt genCodeExpr(Assign);

private:
    CGStmt code;
};

}  // namespace codegen
}  // namespace gern

#endif