#pragma once

namespace gern {
namespace codegen {

struct Literal;
struct Type;
struct VarDecl;
struct DeclFunc;
struct Scope;
struct Block;
struct EscapeCGExpr;
struct EscapeCGStmt;
struct VarAssign;
struct Var;
struct Call;
struct VoidCall;
struct KernelLaunch;
struct For;
struct BlankLine;
struct Lt;
struct Gt;
struct Eq;
struct Neq;
struct Lte;
struct Gte;
struct And;
struct Or;
struct Add;
struct Mod;
struct Div;
struct Mul;
struct Sub;
struct MetaData;
struct Cast;
struct Deref;

/// Extend this class to visit every node in the IR.
class CGVisitorStrict {
public:
    virtual ~CGVisitorStrict() {
    }
    virtual void visit(const Literal *) = 0;
    virtual void visit(const Scope *) = 0;
    virtual void visit(const Type *) = 0;
    virtual void visit(const VarDecl *) = 0;
    virtual void visit(const DeclFunc *) = 0;
    virtual void visit(const Block *) = 0;
    virtual void visit(const EscapeCGExpr *) = 0;
    virtual void visit(const EscapeCGStmt *) = 0;
    virtual void visit(const VarAssign *) = 0;
    virtual void visit(const Var *) = 0;
    virtual void visit(const Call *) = 0;
    virtual void visit(const VoidCall *) = 0;
    virtual void visit(const KernelLaunch *) = 0;
    virtual void visit(const For *) = 0;
    virtual void visit(const BlankLine *) = 0;
    virtual void visit(const Lt *) = 0;
    virtual void visit(const Gt *) = 0;
    virtual void visit(const Eq *) = 0;
    virtual void visit(const Neq *) = 0;
    virtual void visit(const Gte *) = 0;
    virtual void visit(const Lte *) = 0;
    virtual void visit(const And *) = 0;
    virtual void visit(const Or *) = 0;
    virtual void visit(const Add *) = 0;
    virtual void visit(const Mul *) = 0;
    virtual void visit(const Div *) = 0;
    virtual void visit(const Mod *) = 0;
    virtual void visit(const Sub *) = 0;
    virtual void visit(const MetaData *) = 0;
    virtual void visit(const Cast *) = 0;
    virtual void visit(const Deref *) = 0;
    virtual void visit(const SpecializedFunction *) = 0;
};

}  // namespace codegen
}  // namespace gern