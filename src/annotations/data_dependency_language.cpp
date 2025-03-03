#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "compose/compose.h"
#include "utils/debug.h"
#include "utils/error.h"
#include "utils/printer.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace gern {

Expr::Expr(uint32_t val)
    : Expr(new const LiteralNode(val)) {
}
Expr::Expr(uint64_t val)
    : Expr(new const LiteralNode(val)) {
}
Expr::Expr(int32_t val)
    : Expr(new const LiteralNode(val)) {
}
Expr::Expr(int64_t val)
    : Expr(new const LiteralNode(val)) {
}
Expr::Expr(double val)
    : Expr(new const LiteralNode(val)) {
}
Expr::Expr(Grid::Dim dim)
    : Expr(new const GridDimNode(dim)) {
}

void Expr::accept(ExprVisitorStrict *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::string Expr::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::string Constraint::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
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

Variable::Variable(const std::string &name, Datatype type, bool const_expr)
    : Expr(new const VariableNode(name,
                                  Grid::Unit::UNDEFINED,
                                  type, const_expr)) {
}

Variable Variable::bind(int64_t val) const {
    return Variable(new const VariableNode(getName(), getBoundUnit(),
                                           Datatype::Int64, true, true, val));
}

bool Variable::isConstExpr() const {
    return getNode(*this)->const_expr;
}

bool Variable::isBound() const {
    return getNode(*this)->bound;
}

std::ostream &operator<<(std::ostream &os, const AbstractDataTypePtr &ads) {
    if (!ads.defined()) {
        return os;
    }
    os << ads.ptr->getName();
    return os;
}

int64_t Variable::getInt64Val() const {
    return getNode(*this)->val;
}

Grid::Unit Variable::getBoundUnit() const {
    return getNode(*this)->p;
}

Datatype Variable::getDatatype() const {
    return getNode(*this)->type;
}

std::string Variable::getName() const {
    return getNode(*this)->name;
}

Datatype Variable::getType() const {
    return getNode(*this)->type;
}

Assign Variable::operator=(const Expr &e) const {
    return Assign(*this, e);
}

Assign Variable::operator+=(const Expr &e) const {
    return Assign(*this, *this + e);
}

ADTMember::ADTMember(const ADTMemberNode *op)
    : Expr(op) {
}

ADTMember::ADTMember(AbstractDataTypePtr ds,
                     const std::string &member,
                     bool const_expr)
    : ADTMember(new ADTMemberNode(ds, member, const_expr)) {
}

AbstractDataTypePtr ADTMember::getDS() const {
    return getNode(*this)->ds;
}

std::string ADTMember::getMember() const {
    return getNode(*this)->member;
}

GridDim::GridDim(const GridDimNode *n)
    : Expr(n) {
}

GridDim::GridDim(const Grid::Dim &dim)
    : GridDim(new const GridDimNode(dim)) {
}

Grid::Dim GridDim::getDim() const {
    return getNode(*this)->dim;
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

std::string Stmt::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::set<Variable> Stmt::getDefinedVariables() const {
    std::set<Variable> vars;
    match(*this, std::function<void(const AssignNode *)>(
                     [&](const AssignNode *op) { vars.insert(to<Variable>(op->a)); }));
    return vars;
}

std::set<Variable> Stmt::getIntervalVariables() const {
    std::set<Variable> vars;
    match(*this,
          std::function<void(const ConsumesForNode *op, Matcher *ctx)>([&](const ConsumesForNode *op,
                                                                           Matcher *ctx) {
              vars.insert(to<Variable>(op->start.getA()));
              ctx->match(op->body);
          }),
          std::function<void(const ComputesForNode *op, Matcher *ctx)>([&](const ComputesForNode *op,
                                                                           Matcher *ctx) {
              vars.insert(to<Variable>(op->start.getA()));
              ctx->match(op->body);
          }));
    return vars;
}

std::map<Variable, Variable> Stmt::getConsumesIntervalAndStepVars() const {
    std::map<Variable, Variable> vars;
    match(*this,
          std::function<void(const ConsumesForNode *op, Matcher *ctx)>([&](const ConsumesForNode *op,
                                                                           Matcher *ctx) {
              vars[to<Variable>(op->start.getA())] = op->step;
              ctx->match(op->body);
          }));
    return vars;
}

std::map<Variable, Variable> Stmt::getComputesIntervalAndStepVars() const {
    std::map<Variable, Variable> vars;
    match(*this,
          std::function<void(const ComputesForNode *op, Matcher *ctx)>([&](const ComputesForNode *op,
                                                                           Matcher *ctx) {
              vars[to<Variable>(op->start.getA())] = op->step;
              ctx->match(op->body);
          }));
    return vars;
}

std::map<Expr, std::tuple<Variable, Expr, Variable>> Stmt::getTileableFields() const {
    std::map<Expr, std::tuple<Variable, Expr, Variable>> tileable;
    match(*this,
          std::function<void(const ComputesForNode *op, Matcher *ctx)>([&](const ComputesForNode *op,
                                                                           Matcher *ctx) {
              tileable[op->parameter] = std::make_tuple(to<Variable>(op->start.getA()), op->start.getB(), op->step);
              ctx->match(op->body);
          }));
    return tileable;
}

std::map<Expr, std::tuple<Variable, Expr, Variable>> Stmt::getReducableFields() const {
    std::map<Expr, std::tuple<Variable, Expr, Variable>> tileable;
    match(*this,
          std::function<void(const ConsumesForNode *op, Matcher *ctx)>([&](const ConsumesForNode *op,
                                                                           Matcher *ctx) {
              tileable[op->parameter] = std::make_tuple(to<Variable>(op->start.getA()), op->start.getB(), op->step);
              ctx->match(op->body);
          }));
    return tileable;
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

DEFINE_BINARY_CONSTRUCTOR(And, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Or, Constraint)

DEFINE_BINARY_CONSTRUCTOR(Eq, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Neq, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Leq, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Geq, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Less, Constraint)
DEFINE_BINARY_CONSTRUCTOR(Greater, Constraint)

DEFINE_BINARY_CONSTRUCTOR(Assign, Stmt)

SubsetObj::SubsetObj(const SubsetNode *n)
    : Stmt(n) {
}

SubsetObj::SubsetObj(AbstractDataTypePtr data,
                     std::vector<Expr> mdFields)
    : SubsetObj(new const SubsetNode(data, mdFields)) {
}

std::vector<Expr> SubsetObj::getFields() const {
    return getNode(*this)->mdFields;
}

AbstractDataTypePtr SubsetObj::getDS() const {
    return getNode(*this)->data;
}

Produces::Produces(const ProducesNode *n)
    : Stmt(n) {
}

Produces Produces::Subset(AbstractDataTypePtr ds, std::vector<Variable> v) {
    return Produces(new ProducesNode(ds, v));
}

std::vector<Variable> Produces::getFieldsAsVars() const {
    return getNode(*this)->getFieldsAsVars();
}

SubsetObj Produces::getSubset() const {
    return getNode(*this)->output;
}

SubsetObjMany::SubsetObjMany(const SubsetObjManyNode *n)
    : ConsumeMany(n) {
}

SubsetObjMany::SubsetObjMany(const std::vector<SubsetObj> &inputs)
    : SubsetObjMany(new const SubsetObjManyNode(inputs)) {
}

Consumes::Consumes(const ConsumesNode *c)
    : Stmt(c) {
}

Consumes::Consumes(SubsetObj s)
    : Consumes(new const SubsetObjManyNode({s})) {
}

Consumes Consumes::Subset(AbstractDataTypePtr ds, std::vector<Expr> fields) {
    return Consumes(SubsetObj(ds, fields));
}

Consumes Consumes::Subsets(ConsumeMany many) {
    return Consumes(getNode(many));
}

ConsumeMany Reduce(Assign start, Expr parameter, Variable step, ConsumeMany body,
                   bool parallel) {
    return ConsumeMany(
        new const ConsumesForNode(start, parameter, step, body, parallel));
}

ConsumeMany Reduce(Assign start, Expr parameter, Variable step, std::vector<SubsetObj> body,
                   bool parallel) {
    return Reduce(start, parameter, step, SubsetObjMany(body), parallel);
}

ConsumeMany Reduce(Assign start, Expr parameter, Variable step, SubsetObj body,
                   bool parallel) {
    return Reduce(start, parameter, step, std::vector<SubsetObj>{body}, parallel);
}

Allocates::Allocates(const AllocatesNode *n)
    : Stmt(n) {
}

Allocates::Allocates(Expr reg, Expr smem)
    : Allocates(new const AllocatesNode(reg, smem)) {
}

Computes::Computes(const ComputesNode *n)
    : Pattern(n) {
}

Computes::Computes(Produces p, Consumes c, Allocates a)
    : Pattern(new const ComputesNode(p, c, a)) {
}

Pattern::Pattern(const PatternNode *p)
    : Stmt(p) {
}

Annotation Pattern::occupies(std::set<Grid::Unit> occupied) const {
    return Annotation(*this, occupied, {});
}

Annotation Pattern::assumes(std::vector<Constraint> constraints) const {
    return annotate(*this).assumes(constraints);
}

std::vector<SubsetObj> Pattern::getInputs() const {
    std::vector<SubsetObj> subset;
    match(*this, std::function<void(const SubsetObjManyNode *)>(
                     [&](const SubsetObjManyNode *op) {
                         subset = op->subsets;
                     }));
    return subset;
}

std::vector<Variable> Pattern::getProducesField() const {
    std::vector<Variable> fields;
    match(*this, std::function<void(const ProducesNode *)>(
                     [&](const ProducesNode *op) {
                         fields = op->getFieldsAsVars();
                     }));
    return fields;
}

std::vector<Expr> Pattern::getRequirement(AbstractDataTypePtr d) const {
    std::vector<Expr> metaFields;
    match(*this, std::function<void(const SubsetNode *)>(
                     [&](const SubsetNode *op) {
                         if (op->data == d) {
                             metaFields = op->mdFields;
                         }
                     }));
    return metaFields;
}

SubsetObj Pattern::getOutput() const {
    SubsetObj subset;
    match(*this, std::function<void(const ProducesNode *)>(
                     [&](const ProducesNode *op) {
                         subset = op->output;
                     }));
    return subset;
}

SubsetObj Pattern::getCorrespondingSubset(AbstractDataTypePtr d) const {
    SubsetObj subset;
    match(*this, std::function<void(const SubsetNode *)>(
                     [&](const SubsetNode *op) {
                         if (op->data == d) {
                             subset = op;
                         }
                     }));
    return subset;
}

Annotation::Annotation(const AnnotationNode *n)
    : Stmt(n) {
}

Annotation::Annotation(Pattern p,
                       std::set<Grid::Unit> occupied,
                       std::vector<Constraint> constraints)
    : Annotation(new const AnnotationNode(p, occupied, constraints)) {
}

Annotation annotate(Pattern p) {
    return Annotation(p, {}, {});
}

Annotation Annotation::assumes(std::vector<Constraint> constraints) const {

    auto annotVars = getVariables(getPattern());
    auto legalDims = getDims(getOccupiedUnits());       // Dimensions of the grid the function can actually control.
    auto current_level = getLevel(getOccupiedUnits());  // Level at which the function is at.

    for (const auto &c : constraints) {
        auto constraintVars = getVariables(c);
        if (!std::includes(annotVars.begin(), annotVars.end(), constraintVars.begin(),
                           constraintVars.end(), std::less<Variable>())) {
            throw error::UserError("Putting constraints on variables that are not "
                                   "present in statement's scope");
        }
        auto constraintDims = getDims(c);
        for (const auto &c_dim : constraintDims) {
            if (!isDimInScope(c_dim, legalDims)) {
                throw error::UserError(" Cannot constrain " +
                                       util::str(c_dim) + " at " +
                                       util::str(current_level));
            }
        }
    }
    return Annotation(getPattern(), getOccupiedUnits(), constraints);
}

Pattern Annotation::getPattern() const {
    return getNode(*this)->p;
}

std::vector<Constraint> Annotation::getConstraints() const {
    return getNode(*this)->constraints;
}

std::set<Grid::Unit> Annotation::getOccupiedUnits() const {
    return getNode(*this)->occupied;
}

Annotation resetUnit(Annotation annot, std::set<Grid::Unit> occupied) {
    return Annotation(annot.getPattern(), occupied, annot.getConstraints());
}

Pattern For(Assign start, Expr parameter, Variable step, Pattern body,
            bool parallel) {
    return Pattern(
        new const ComputesForNode(start, parameter, step, body, parallel));
}

Pattern For(Assign start, Expr parameter, Variable step,
            Produces produces, Consumes consumes,
            bool parallel) {
    return For(start, parameter, step, Computes(produces, consumes), parallel);
}

std::string AbstractDataTypePtr::getName() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getName();
}

std::string AbstractDataTypePtr::getType() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getType();
}

FunctionSignature AbstractDataTypePtr::getAllocateFunction() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getAllocateFunction();
}

FunctionSignature AbstractDataTypePtr::getQueryFunction() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getQueryFunction();
}

FunctionSignature AbstractDataTypePtr::getInsertFunction() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getInsertFunction();
}

std::vector<Variable> AbstractDataTypePtr::getFields() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->getFields();
}

bool AbstractDataTypePtr::freeQuery() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->freeQuery();
}

bool AbstractDataTypePtr::insertQuery() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->insertQuery();
}

bool AbstractDataTypePtr::freeAlloc() const {
    if (!defined()) {
        throw error::InternalError("Deref null!");
    }
    return ptr->freeAlloc();
}

ADTMember AbstractDataTypePtr::operator[](std::string member) const {
    return ADTMember(*this, member, false);
}

std::string AbstractDataTypePtr::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

}  // namespace gern