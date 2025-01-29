#pragma once

// Impl more or less from TACO impl
#include "annotations/datatypes.h"
#include "utils/uncopyable.h"
#include <any>
#include <cassert>

namespace gern {
namespace codegen {

class CGVisitorStrict;

enum class CGNodeType {
    Literal,
    Var,
    Neg,
    Sqrt,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Rem,
    Min,
    Max,
    BitAnd,
    BitOr,
    Not,
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    And,
    Or,
    BinOp,
    Cast,
    CustomCast,
    Deref,
    Call,
    IfThenElse,
    Case,
    Switch,
    Load,
    Malloc,
    Sizeof,
    Store,
    For,
    While,
    Block,
    Scope,
    Function,
    VarDecl,
    VarAssign,
    Yield,
    Allocate,
    Free,
    Comment,
    BlankLine,
    Print,
    Continue,
    Sort,
    Break,
    VoidCall,
    KernelLaunch,
    DeclObject,
    CustomObject,
    RawString,
    DeclFunc,
    Type,
    Assign,
    // Escape helps us insert arbitrary strings
    // into the generated code (no checks)
    EscapeCGExpr,
    EscapeCGStmt,
    MetaData,
};

/** Base class for backend CG */
struct CGNode : private util::Uncopyable {
    CGNode() {
    }
    virtual ~CGNode() {
    }
    virtual void accept(CGVisitorStrict *v) const = 0;

    /** Each CGNode subclasses carries a unique pointer we use to determine
     * its node type, because compiler RTTI sucks.
     */
    virtual CGNodeType type_info() const = 0;

    mutable long ref = 0;
    friend void acquire(const CGNode *node) {
        ++(node->ref);
    }
    friend void release(const CGNode *node) {
        if (--(node->ref) == 0) {
            delete node;
        }
    }
};

/** Base class for statements. */
struct BaseCGStmtNode : public CGNode {};

/** Base class for expression nodes, which have a type. */
struct BaseCGExprNode : public CGNode {
    Datatype type = Float();
};

/** Use the "curiously recurring template pattern" from Halide
 * to avoid duplicated code in CG nodes.  This provides the type
 * info for each class (and will handle visitor accept methods as
 * well).
 */
template<typename T>
struct CGExprNode : public BaseCGExprNode {
    virtual ~CGExprNode() = default;
    void accept(CGVisitorStrict *v) const;
    virtual CGNodeType type_info() const {
        return T::_type_info;
    }
};

template<typename T>
struct CGStmtNode : public BaseCGStmtNode {
    virtual ~CGStmtNode() = default;
    void accept(CGVisitorStrict *v) const;
    virtual CGNodeType type_info() const {
        return T::_type_info;
    }
};

/** CG nodes are passed around using opaque handles.  This class
 * handles type conversion, and will handle visitors.
 */
struct CGHandle : public util::IntrusivePtr<const CGNode> {
    CGHandle()
        : util::IntrusivePtr<const CGNode>() {
    }
    CGHandle(const CGNode *p)
        : util::IntrusivePtr<const CGNode>(p) {
    }

    /** Cast this CG node to its actual type. */
    template<typename T>
    const T *as() const {
        if (ptr && ptr->type_info() == T::_type_info) {
            return (const T *)ptr;
        } else {
            return nullptr;
        }
    }

    /** Dispatch to the corresponding visitor method */
    void accept(CGVisitorStrict *v) const {
        ptr->accept(v);
    }
};

/** An expression. */
class CGExpr : public CGHandle {
public:
    CGExpr()
        : CGHandle() {
    }

    CGExpr(bool);
    CGExpr(int8_t);
    CGExpr(int16_t);
    CGExpr(int32_t);
    CGExpr(int64_t);
    CGExpr(uint8_t);
    CGExpr(uint16_t);
    CGExpr(uint32_t);
    CGExpr(uint64_t);
    CGExpr(float);
    CGExpr(double);
    CGExpr(std::complex<float>);
    CGExpr(std::complex<double>);

    CGExpr(const BaseCGExprNode *expr)
        : CGHandle(expr) {
    }

    std::string str() const;

    /** Get the type of this expression node */
    Datatype type() const {
        return ((const BaseCGExprNode *)ptr)->type;
    }
};

CGExpr operator+(const CGExpr &, const CGExpr &);
CGExpr operator-(const CGExpr &, const CGExpr &);
CGExpr operator*(const CGExpr &, const CGExpr &);
CGExpr operator/(const CGExpr &, const CGExpr &);
CGExpr operator==(const CGExpr &, const CGExpr &);
CGExpr operator!=(const CGExpr &, const CGExpr &);
CGExpr operator<=(const CGExpr &, const CGExpr &);
CGExpr operator>=(const CGExpr &, const CGExpr &);
CGExpr operator<(const CGExpr &, const CGExpr &);
CGExpr operator>(const CGExpr &, const CGExpr &);
CGExpr operator&&(const CGExpr &, const CGExpr &);
CGExpr operator||(const CGExpr &, const CGExpr &);

/** This is a custom comparator that allows
 * CGExprs to be used in a map.  Inspired by Halide.
 */
class CGExprCompare {
public:
    bool operator()(CGExpr a, CGExpr b) const {
        return a.ptr < b.ptr;
    }
};

/** A statement. */
class CGStmt : public CGHandle {
public:
    CGStmt()
        : CGHandle() {
    }
    CGStmt(const BaseCGStmtNode *stmt)
        : CGHandle(stmt) {
    }
};

std::ostream &operator<<(std::ostream &os, const CGStmt &);
std::ostream &operator<<(std::ostream &os, const CGExpr &);

// TODO: CHANGE TO USE ANY
/** A literal. */
struct Literal : public CGExprNode<Literal> {
    std::any val;
    Datatype type;

    static CGExpr make(std::any val, Datatype type) {
        Literal *lit = new Literal;
        lit->val = val;
        lit->type = type;
        return lit;
    }

    template<typename T>
    static CGExpr make(T val) {
        return make(val, gern::type<T>());
    }

    /// Returns a zero literal of the given type.
    static CGExpr zero(Datatype datatype);

    template<typename T>
    T getVal() const {
        return std::any_cast<T>(val);
    }

    // bool getBoolValue() const;
    // int64_t getIntValue() const;
    // uint64_t getUIntValue() const;
    // double getFloatValue() const;
    // std::complex<double> getComplexValue() const;

    static const CGNodeType _type_info = CGNodeType::Literal;
};

struct Type : public CGExprNode<Type> {
    Type(const std::string &type_name)
        : type_name(type_name) {
    }
    static CGExpr make(const std::string &type_name) {
        return new Type(type_name);
    }
    std::string type_name;
    static const CGNodeType _type_info = CGNodeType::Type;
};

struct DeclProperties {
    bool is_const = false;  // Is a const decl.
    int num_ref = 0;        // Declare reference.
    int num_ptr = 0;        // Track that this is declared.
};

struct VarDecl : public CGExprNode<VarDecl> {
    static CGExpr make(CGExpr type, const std::string &var_name,
                       DeclProperties properties) {
        VarDecl *var = new VarDecl;
        var->type = type;
        var->var_name = var_name;
        var->properties = properties;
        return var;
    }

    static CGExpr make(CGExpr type, const std::string &var_name,
                       bool is_const = false,
                       int num_ref = 0, int num_ptr = 0) {
        VarDecl *var = new VarDecl;
        var->type = type;
        var->var_name = var_name;
        var->properties.is_const = is_const;
        var->properties.num_ref = num_ref;
        var->properties.num_ptr = num_ptr;
        return var;
    }

    CGExpr type;
    std::string var_name;
    DeclProperties properties;

    static const CGNodeType _type_info = CGNodeType::VarDecl;
};

struct Var : public CGExprNode<Var> {
    static CGExpr make(std::string name) {
        Var *v = new Var;
        v->name = name;
        return v;
    }
    std::string name;
    static const CGNodeType _type_info = CGNodeType::Var;
};

struct MetaData : public CGExprNode<MetaData> {
    static CGExpr make(CGExpr var, std::string field) {
        MetaData *md = new MetaData;
        md->var = var;
        md->field = field;
        return md;
    }
    std::string field;
    CGExpr var;
    static const CGNodeType _type_info = CGNodeType::MetaData;
};

struct EscapeCGExpr : public CGExprNode<EscapeCGExpr> {
    static CGExpr make(std::string code) {
        EscapeCGExpr *e = new EscapeCGExpr;
        e->code = code;
        return e;
    }
    std::string code;
    static const CGNodeType _type_info = CGNodeType::EscapeCGExpr;
};

struct Gt : CGExprNode<Gt> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Gt *node = new Gt;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Gt;
};

struct Lt : CGExprNode<Lt> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Lt *node = new Lt;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Lt;
};

struct Eq : CGExprNode<Eq> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Eq *node = new Eq;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Eq;
};

struct Neq : CGExprNode<Neq> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Neq *node = new Neq;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Neq;
};

struct Lte : CGExprNode<Lte> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Lte *node = new Lte;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Lte;
};

struct Gte : CGExprNode<Gte> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Gte *node = new Gte;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Gte;
};

struct And : CGExprNode<And> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Gte *node = new Gte;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Add;
};

struct Or : CGExprNode<Or> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Gte *node = new Gte;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Or;
};

struct Add : CGExprNode<Add> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Add *node = new Add;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Add;
};

struct Sub : CGExprNode<Sub> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Sub *node = new Sub;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Sub;
};

struct Mul : CGExprNode<Mul> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Mul *node = new Mul;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Mul;
};

struct Div : CGExprNode<Div> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Div *node = new Div;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Div;
};

struct Mod : CGExprNode<Mod> {
    CGExpr a;
    CGExpr b;

    static CGExpr make(CGExpr a, CGExpr b) {
        Mod *node = new Mod;
        node->a = a;
        node->b = b;
        return node;
    }

    static const CGNodeType _type_info = CGNodeType::Mod;
};

struct EscapeCGStmt : public CGStmtNode<EscapeCGStmt> {
    static CGStmt make(std::string code) {
        EscapeCGStmt *e = new EscapeCGStmt;
        e->code = code;
        return e;
    }
    std::string code;
    static const CGNodeType _type_info = CGNodeType::EscapeCGStmt;
};

/** A literal. */
struct Scope : public CGStmtNode<Scope> {
    Scope(CGStmt stmt)
        : stmt(stmt) {};
    CGStmt getCGStmt() {
        return stmt;
    }
    static CGStmt make(CGStmt stmt) {
        return new Scope(stmt);
    }
    CGStmt stmt;
    static const CGNodeType _type_info = CGNodeType::Scope;
};

struct DeclFunc : public CGStmtNode<DeclFunc> {
    std::string name;
    CGExpr return_type;
    std::vector<CGExpr> args;
    CGStmt body;
    bool device;
    std::vector<CGExpr> template_args;

    static CGStmt make(std::string name, CGExpr return_type, std::vector<CGExpr> args,
                       CGStmt body, bool device = false, std::vector<CGExpr> template_args = {}) {
        DeclFunc *declFunc = new DeclFunc;
        declFunc->name = name;
        declFunc->return_type = return_type;
        declFunc->args = args;
        declFunc->device = device;
        declFunc->body = Scope::make(body);
        declFunc->template_args = template_args;
        return declFunc;
    }

    static const CGNodeType _type_info = CGNodeType::DeclFunc;
};

struct Block : public CGStmtNode<Block> {
    std::vector<CGStmt> stmts;
    static CGStmt make(std::vector<CGStmt> stmts) {
        Block *blk = new Block;
        blk->stmts = stmts;
        return blk;
    }

    template<typename... CGStmts>
    static CGStmt make(const CGStmts &...stmts) {
        return make({stmts...});
    }

    static const CGNodeType _type_info = CGNodeType::Block;
};

enum AssignOp { AddEQ,
                SubEQ,
                EQ,
                MulEQ };

struct VarAssign : public CGStmtNode<VarAssign> {
    CGExpr lhs;
    CGExpr rhs;
    AssignOp op;

    static CGStmt make(CGExpr lhs, CGExpr rhs, AssignOp op = EQ) {
        VarAssign *a = new VarAssign;
        a->rhs = rhs;
        a->lhs = lhs;
        a->op = op;
        return a;
    }
    static const CGNodeType _type_info = CGNodeType::VarAssign;
};

struct Call : public CGExprNode<Call> {
    std::vector<CGExpr> arg;
    std::string name;
    std::vector<CGExpr> template_args;

    static CGExpr make(std::string name, std::vector<CGExpr> arg,
                       std::vector<CGExpr> template_args = {}) {
        Call *call = new Call;
        call->name = name;
        call->arg = arg;
        call->template_args = template_args;
        return call;
    }

    static const CGNodeType _type_info = CGNodeType::Call;
};

struct Cast : public CGExprNode<Cast> {
    CGExpr to_cast;
    CGExpr cast_type;

    static CGExpr make(CGExpr cast_type, CGExpr to_cast) {
        Cast *cast = new Cast;
        cast->to_cast = to_cast;
        cast->cast_type = cast_type;
        return cast;
    }

    static const CGNodeType _type_info = CGNodeType::Cast;
};

struct Deref : public CGExprNode<Deref> {
    CGExpr expr;

    static CGExpr make(CGExpr expr) {
        Deref *deref = new Deref;
        deref->expr = expr;
        return deref;
    }

    static const CGNodeType _type_info = CGNodeType::Deref;
};

struct For : public CGStmtNode<For> {
    CGStmt start;
    CGExpr cond;
    CGStmt step;
    CGStmt body;
    bool parallel;

    static CGStmt make(CGStmt start, CGExpr cond, CGStmt step, CGStmt body,
                       bool parallel = false) {
        For *f = new For;
        f->start = start;
        f->cond = cond;
        f->step = step;
        f->body = Scope::make(body);
        f->parallel = parallel;
        return f;
    }

    static const CGNodeType _type_info = CGNodeType::For;
};

struct BlankLine : public CGStmtNode<BlankLine> {
    static CGStmt make() {
        BlankLine *b = new BlankLine;
        return b;
    }
    static const CGNodeType _type_info = CGNodeType::BlankLine;
};

struct VoidCall : public CGStmtNode<VoidCall> {
    // Should be of type call, just wrapping in stmt
    CGExpr func;

    static CGStmt make(CGExpr func) {
        VoidCall *call = new VoidCall;
        call->func = func;
        return call;
    }
    static const CGNodeType _type_info = CGNodeType::VoidCall;
};

struct KernelLaunch : public CGStmtNode<KernelLaunch> {
    // Should be of type call, just wrapping in stmt
    std::string name;
    std::vector<CGExpr> args;
    std::vector<CGExpr> template_args;
    CGExpr grid;
    CGExpr block;

    static CGStmt make(std::string name,
                       std::vector<CGExpr> args,
                       std::vector<CGExpr> template_args,
                       CGExpr grid,
                       CGExpr block) {

        KernelLaunch *call = new KernelLaunch;
        call->name = name;
        call->args = args;
        call->template_args = template_args;
        call->grid = grid;
        call->block = block;
        return call;
    }
    static const CGNodeType _type_info = CGNodeType::KernelLaunch;
};

template<typename E>
inline bool isa(CGExpr e) {
    return e.defined() && dynamic_cast<const E *>(e.ptr) != nullptr;
}

template<typename S>
inline bool isa(CGStmt s) {
    return s.defined() && dynamic_cast<const S *>(s.ptr) != nullptr;
}

template<typename E>
inline const E *to(CGExpr e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e.ptr);
}

template<typename S>
inline const S *to(CGStmt s) {
    assert(isa<S>(s));
    return static_cast<const S *>(s.ptr);
}

}  // namespace codegen
}  // namespace gern