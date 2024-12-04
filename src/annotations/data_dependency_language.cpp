#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <set>

namespace gern {

Expr::Expr(uint8_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(uint16_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(uint32_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(uint64_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int8_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int16_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int32_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int64_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(float val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(double val)
    : Expr(new LiteralNode(val)) {
}

void Expr::accept(ExprVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void Constraint::accept(ConstraintVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

void Stmt::accept(StmtVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

Variable::Variable(const VariableNode *n)
    : Expr(n) {
}

Variable::Variable(const std::string &name)
    : Expr(new const VariableNode(name)) {
}

std::ostream &operator<<(std::ostream &os, const Expr &e) {
    Printer p{os};
    p.visit(e);
    return os;
}

std::ostream &operator<<(std::ostream &os, const Constraint &c) {
    Printer p{os};
    p.visit(c);
    return os;
}

#define DEFINE_BINARY_OPERATOR(CLASS_NAME, OPERATOR, NODE)       \
    CLASS_NAME operator OPERATOR(const Expr &a, const Expr &b) { \
        return CLASS_NAME(a, b);                                 \
    }

DEFINE_BINARY_OPERATOR(Add, +, Expr)
DEFINE_BINARY_OPERATOR(Sub, -, Expr)
DEFINE_BINARY_OPERATOR(Mul, *, Expr)
DEFINE_BINARY_OPERATOR(Div, /, Expr)
DEFINE_BINARY_OPERATOR(Mod, %, Expr)

DEFINE_BINARY_OPERATOR(Eq, ==, Constraint)
DEFINE_BINARY_OPERATOR(Neq, !=, Constraint)
DEFINE_BINARY_OPERATOR(Leq, <=, Constraint)
DEFINE_BINARY_OPERATOR(Geq, >=, Constraint)
DEFINE_BINARY_OPERATOR(Less, <, Constraint)
DEFINE_BINARY_OPERATOR(Greater, >, Constraint)
DEFINE_BINARY_OPERATOR(And, &&, Constraint)
DEFINE_BINARY_OPERATOR(Or, ||, Constraint)

std::ostream &operator<<(std::ostream &os, const Stmt &s) {
    Printer p{os};
    p.visit(s);
    return os;
}

template<typename T>
std::set<const VariableNode *> getVariables(T annot) {
    std::set<const VariableNode *> vars;
    match(annot, std::function<void(const VariableNode *)>(
                     [&](const VariableNode *op) { vars.insert(op); }));
    return vars;
}

Stmt Stmt::where(Constraint constraint) {
    auto stmtVars = getVariables(*this);
    auto constraintVars = getVariables(constraint);
    if (!std::includes(stmtVars.begin(), stmtVars.end(), constraintVars.begin(),
                       constraintVars.end())) {
        throw error::UserError("Putting constraints on variables that are not "
                               "present in statement's scope");
    }
    return Stmt(ptr, constraint);
}

Stmt Stmt::replaceVariables(std::map<Variable, Variable> rw_vars) const {
    struct rewriteVar : public Rewriter {
        rewriteVar(std::map<Variable, Variable> rw_vars)
            : rw_vars(rw_vars) {
        }
        using Rewriter::rewrite;

        void visit(const VariableNode *op) {
            if (rw_vars.find(op) != rw_vars.end()) {
                expr = rw_vars[op];
            } else {
                expr = op;
            }
        }
        std::map<Variable, Variable> rw_vars;
    };
    rewriteVar rw{rw_vars};
    return rw.rewrite(*this);
}

Stmt Stmt::replaceDSArgs(std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds) const {
    struct rewriteDS : public Rewriter {
        rewriteDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds)
            : rw_ds(rw_ds) {
        }
        using Rewriter::rewrite;

        void visit(const SubsetNode *op) {
            if (rw_ds.find(op->data) != rw_ds.end()) {
                stmt = Subset(rw_ds[op->data], op->mdFields);
            } else {
                stmt = Subset(op->data, op->mdFields);
            }
        }
        std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds;
    };
    rewriteDS rw{rw_ds};
    return rw.rewrite(*this);
}

#define DEFINE_BINARY_CONSTRUCTOR(CLASS_NAME, NODE)               \
    CLASS_NAME::CLASS_NAME(const CLASS_NAME##Node *n) : NODE(n) { \
    }                                                             \
    CLASS_NAME::CLASS_NAME(Expr a, Expr b)                        \
        : NODE(new const CLASS_NAME##Node(a, b)) {                \
    }                                                             \
    Expr CLASS_NAME::getA() const {                               \
        return getNode(*this)->a;                                 \
    }                                                             \
    Expr CLASS_NAME::getB() const {                               \
        return getNode(*this)->b;                                 \
    }

DEFINE_BINARY_CONSTRUCTOR(Add, Expr)
DEFINE_BINARY_CONSTRUCTOR(Sub, Expr)
DEFINE_BINARY_CONSTRUCTOR(Div, Expr)
DEFINE_BINARY_CONSTRUCTOR(Mod, Expr)
DEFINE_BINARY_CONSTRUCTOR(Mul, Expr)

DEFINE_BINARY_CONSTRUCTOR(And, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Or, Constraint);

DEFINE_BINARY_CONSTRUCTOR(Eq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Neq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Leq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Geq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Less, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Greater, Constraint);

Subset::Subset(const SubsetNode *n)
    : Stmt(n) {
}

Subset::Subset(AbstractDataTypePtr data,
               std::vector<Expr> mdFields)
    : Stmt(new const SubsetNode(data, mdFields)) {
}

Subsets::Subsets(const std::vector<Subset> &inputs)
    : ConsumeMany(new const SubsetsNode(inputs)) {
}

Produces::Produces(const ProducesNode *n)
    : Stmt(n) {
}

Produces::Produces(Subset s)
    : Stmt(new const ProducesNode(s)) {
}

Subset Produces::getSubset() {
    return getNode(*this)->output;
}

Subsets::Subsets(const SubsetsNode *n)
    : ConsumeMany(n) {
}

Consumes::Consumes(const ConsumesNode *c)
    : Stmt(c) {
}

Consumes::Consumes(Subset s)
    : Consumes(new const SubsetsNode({s})) {
}

ConsumeMany For(Variable v, Expr start, Expr end, Expr step, ConsumeMany body,
                bool parallel) {
    return ConsumeMany(
        new const ConsumesForNode(v, start, end, step, body, parallel));
}

Allocates::Allocates(const AllocatesNode *n)
    : Stmt(n) {
}

Allocates::Allocates(Expr reg, Expr smem)
    : Stmt(new const AllocatesNode(reg, smem)) {
}

Computes::Computes(Produces p, Consumes c, Allocates a)
    : Pattern(new const ComputesNode(p, c, a)) {
}

Pattern::Pattern(const PatternNode *p)
    : Stmt(p) {
}

Pattern For(Variable v, Expr start, Expr end, Expr step, Pattern body,
            bool parallel) {
    return Pattern(
        new const ComputesForNode(v, start, end, step, body, parallel));
}

}  // namespace gern