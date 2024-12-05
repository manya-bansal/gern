#ifndef CODEGEN_VISITOR_H
#define CODEGEN_VISITOR_H

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
struct For;
struct BlankLine;
struct Lt;
struct Gt;
struct Eq;
struct Neq;
struct Lte;
struct Gte;
struct Add;
struct Mod;
struct Div;
struct Mul;
struct Sub;
struct MetaData;

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
    virtual void visit(const For *) = 0;
    virtual void visit(const BlankLine *) = 0;
    virtual void visit(const Lt *) = 0;
    virtual void visit(const Gt *) = 0;
    virtual void visit(const Eq *) = 0;
    virtual void visit(const Neq *) = 0;
    virtual void visit(const Gte *) = 0;
    virtual void visit(const Lte *) = 0;
    virtual void visit(const Add *) = 0;
    virtual void visit(const Mul *) = 0;
    virtual void visit(const Div *) = 0;
    virtual void visit(const Mod *) = 0;
    virtual void visit(const Sub *) = 0;
    virtual void visit(const MetaData *) = 0;
    // virtual void visit(const CilkFor*) = 0;
};

/// Extend this class to visit some nodes in the IR.
class CGVisitor : public CGVisitorStrict {
public:
    virtual ~CGVisitor() {
    }
    using CGVisitorStrict::visit;
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
    virtual void visit(const For *);
    virtual void visit(const BlankLine *);
    virtual void visit(const Lt *);
    virtual void visit(const Gt *);
    virtual void visit(const Eq *);
    virtual void visit(const Neq *);
    virtual void visit(const Gte *);
    virtual void visit(const Lte *);
    virtual void visit(const Add *);
    virtual void visit(const Mul *);
    virtual void visit(const Div *);
    virtual void visit(const Mod *);
    virtual void visit(const Sub *);
    virtual void visit(const MetaData *);
    // virtual void visit(const CilkFor*);
};

}  // namespace codegen
}  // namespace gern

#endif