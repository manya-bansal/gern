#include "codegen/codegen_ir.h"
#include "codegen/codegen_printer.h"
#include "codegen/codegen_visitor.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

// class CGExpr
CGExpr::CGExpr(bool n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(int8_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(int16_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(int32_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(int64_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(uint8_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(uint16_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(uint32_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(uint64_t n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(float n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(double n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(std::complex<float> n)
    : CGHandle(Literal::make(n)) {
}

CGExpr::CGExpr(std::complex<double> n)
    : CGHandle(Literal::make(n)) {
}

CGExpr Literal::zero(Datatype datatype) {
    CGExpr zero;
    switch (datatype.getKind()) {
    case Datatype::Bool:
        zero = Literal::make(false);
        break;
    case Datatype::UInt8:
        zero = Literal::make((uint8_t)0);
        break;
    case Datatype::UInt16:
        zero = Literal::make((uint16_t)0);
        break;
    case Datatype::UInt32:
        zero = Literal::make((uint32_t)0);
        break;
    case Datatype::UInt64:
        zero = Literal::make((uint64_t)0);
        break;
    case Datatype::Int8:
        zero = Literal::make((int8_t)0);
        break;
    case Datatype::Int16:
        zero = Literal::make((int16_t)0);
        break;
    case Datatype::Int32:
        zero = Literal::make((int32_t)0);
        break;
    case Datatype::Int64:
        zero = Literal::make((int64_t)0);
        break;

    case Datatype::Float32:
        zero = Literal::make((float)0.0);
        break;
    case Datatype::Float64:
        zero = Literal::make((double)0.0);
        break;
    case Datatype::Complex64:
        zero = Literal::make(std::complex<float>());
        break;
    case Datatype::Complex128:
        zero = Literal::make(std::complex<double>());
        break;
    case Datatype::Undefined:
        throw error::InternalError("Unreachable");
        break;
    }
    return zero;
}

// printing methods
std::ostream &operator<<(std::ostream &os, const CGStmt &stmt) {
    if (!stmt.defined())
        return os << "CGStmt()" << std::endl;
    CGPrinter printer(os);
    stmt.accept(&printer);
    return os;
}

std::ostream &operator<<(std::ostream &os, const CGExpr &expr) {
    if (!expr.defined())
        return os << "CGExpr()";
    CGPrinter printer(os);
    expr.accept(&printer);
    return os;
}

std::string CGExpr::str() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}

#define DEFINE_BINARY_OPERATOR(OPERATOR, NODE)                   \
    CGExpr operator OPERATOR(const CGExpr &a, const CGExpr &b) { \
        return NODE::make(a, b);                                 \
    }

DEFINE_BINARY_OPERATOR(+, Add)
DEFINE_BINARY_OPERATOR(-, Sub)
DEFINE_BINARY_OPERATOR(*, Mul)
DEFINE_BINARY_OPERATOR(/, Div)
DEFINE_BINARY_OPERATOR(==, Eq)
DEFINE_BINARY_OPERATOR(!=, Neq)
DEFINE_BINARY_OPERATOR(<=, Lte)
DEFINE_BINARY_OPERATOR(>=, Gte)
DEFINE_BINARY_OPERATOR(<, Lt)
DEFINE_BINARY_OPERATOR(>, Gt)
DEFINE_BINARY_OPERATOR(&&, And)
DEFINE_BINARY_OPERATOR(||, Or)

template<>
void CGExprNode<Literal>::accept(CGVisitorStrict *v) const {
    v->visit((const Literal *)this);
}

template<>
void CGExprNode<Type>::accept(CGVisitorStrict *v) const {
    v->visit((const Type *)this);
}

template<>
void CGExprNode<VarDecl>::accept(CGVisitorStrict *v) const {
    v->visit((const VarDecl *)this);
}

template<>
void CGExprNode<EscapeCGExpr>::accept(CGVisitorStrict *v) const {
    v->visit((const EscapeCGExpr *)this);
}

template<>
void CGExprNode<Var>::accept(CGVisitorStrict *v) const {
    v->visit((const Var *)this);
}

template<>
void CGExprNode<Call>::accept(CGVisitorStrict *v) const {
    v->visit((const Call *)this);
}

template<>
void CGExprNode<Gt>::accept(CGVisitorStrict *v) const {
    v->visit((const Gt *)this);
}

template<>
void CGExprNode<Lt>::accept(CGVisitorStrict *v) const {
    v->visit((const Lt *)this);
}

template<>
void CGExprNode<Eq>::accept(CGVisitorStrict *v) const {
    v->visit((const Eq *)this);
}

template<>
void CGExprNode<Neq>::accept(CGVisitorStrict *v) const {
    v->visit((const Neq *)this);
}

template<>
void CGExprNode<Lte>::accept(CGVisitorStrict *v) const {
    v->visit((const Lte *)this);
}

template<>
void CGExprNode<Gte>::accept(CGVisitorStrict *v) const {
    v->visit((const Gte *)this);
}

template<>
void CGExprNode<Add>::accept(CGVisitorStrict *v) const {
    v->visit((const Add *)this);
}

template<>
void CGExprNode<Sub>::accept(CGVisitorStrict *v) const {
    v->visit((const Sub *)this);
}

template<>
void CGExprNode<Mul>::accept(CGVisitorStrict *v) const {
    v->visit((const Mul *)this);
}

template<>
void CGExprNode<Div>::accept(CGVisitorStrict *v) const {
    v->visit((const Div *)this);
}

template<>
void CGExprNode<Mod>::accept(CGVisitorStrict *v) const {
    v->visit((const Mod *)this);
}

template<>
void CGExprNode<MetaData>::accept(CGVisitorStrict *v) const {
    v->visit((const MetaData *)this);
}

template<>
void CGExprNode<Cast>::accept(CGVisitorStrict *v) const {
    v->visit((const Cast *)this);
}

template<>
void CGExprNode<Deref>::accept(CGVisitorStrict *v) const {
    v->visit((const Deref *)this);
}

template<>
void CGStmtNode<VarAssign>::accept(CGVisitorStrict *v) const {
    v->visit((const VarAssign *)this);
}

template<>
void CGStmtNode<For>::accept(CGVisitorStrict *v) const {
    v->visit((const For *)this);
}

template<>
void CGStmtNode<BlankLine>::accept(CGVisitorStrict *v) const {
    v->visit((const BlankLine *)this);
}

template<>
void CGStmtNode<VoidCall>::accept(CGVisitorStrict *v) const {
    v->visit((const VoidCall *)this);
}

template<>
void CGExprNode<KernelLaunch>::accept(CGVisitorStrict *v) const {
    v->visit((const KernelLaunch *)this);
}

template<>
void CGStmtNode<DeclFunc>::accept(CGVisitorStrict *v) const {
    v->visit((const DeclFunc *)this);
}

template<>
void CGStmtNode<EscapeCGStmt>::accept(CGVisitorStrict *v) const {
    v->visit((const EscapeCGStmt *)this);
}

template<>
void CGStmtNode<Block>::accept(CGVisitorStrict *v) const {
    v->visit((const Block *)this);
}

template<>
void CGStmtNode<Scope>::accept(CGVisitorStrict *v) const {
    v->visit((const Scope *)this);
}

template<>
void CGExprNode<SpecializedFunction>::accept(CGVisitorStrict *v) const {
    v->visit((const SpecializedFunction *)this);
}

// template <>
// void CGStmtNode<CilkFor>::accept(CGVisitorStrict* v)
//     const { v->visit((const CilkFor*)this); }

}  // namespace codegen
}  // namespace gern