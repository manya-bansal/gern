#pragma once

#include "codegen/codegen_ir.h"
#include "codegen/codegen_visitor.h"

namespace gern {
namespace codegen {

class CGPrinter : public CGVisitorStrict {
public:
    CGPrinter(std::ostream &os)
        : os(os) {
    }
    virtual ~CGPrinter() {
    }

    void print(CGStmt);

    virtual void visit(const Literal *);
    virtual void visit(const Scope *);
    virtual void visit(const Type *);
    virtual void visit(const VarDecl *);
    virtual void visit(const DeclFunc *);
    virtual void visit(const Block *);
    virtual void visit(const EscapeCGExpr *);
    virtual void visit(const EscapeCGStmt *);
    virtual void visit(const VarAssign *);
    virtual void visit(const Var *);
    virtual void visit(const Call *);
    virtual void visit(const VoidCall *);
    virtual void visit(const KernelLaunch *);
    virtual void visit(const BlankLine *);
    virtual void visit(const For *);
    virtual void visit(const Lt *);
    virtual void visit(const Gt *);
    virtual void visit(const Eq *);
    virtual void visit(const Neq *);
    virtual void visit(const Gte *);
    virtual void visit(const Lte *);
    virtual void visit(const And *);
    virtual void visit(const Or *);
    virtual void visit(const Add *);
    virtual void visit(const Mul *);
    virtual void visit(const Div *);
    virtual void visit(const Mod *);
    virtual void visit(const Sub *);
    virtual void visit(const MetaData *);
    virtual void visit(const Cast *);
    virtual void visit(const Deref *);

private:
    std::ostream &os;
    void doIdent();
};

}  // namespace codegen
}  // namespace gern