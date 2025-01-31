#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/compose.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace gern {

Expr::Expr(uint32_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(uint64_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int32_t val)
    : Expr(new LiteralNode(val)) {
}
Expr::Expr(int64_t val)
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

Variable::Variable(const std::string &name)
    : Expr(new const VariableNode(name)) {
}

Variable::Variable(const std::string &name, bool const_expr)
    : Expr(new const VariableNode(name,
                                  Grid::Property::UNDEFINED,
                                  Datatype::Int64, const_expr)) {
}

Variable Variable::bindToGrid(const Grid::Property &p) const {
    return Variable(new const VariableNode(getName(), p));
}

Variable Variable::bindToInt64(int64_t val) const {
    return Variable(new const VariableNode(getName(), getBoundProperty(),
                                           Datatype::Int64, true, true, val));
}

bool Variable::isBoundToGrid() const {
    return isGridPropertySet(getBoundProperty());
}

bool Variable::isConstExpr() const {
    return getNode(*this)->const_expr;
}

bool Variable::isBoundToInt64() const {
    return getNode(*this)->bound;
}

bool Variable::isBound() const {
    return (isBoundToGrid() || isBoundToInt64());
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

Grid::Property Variable::getBoundProperty() const {
    return getNode(*this)->p;
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

ADTMember::ADTMember(AbstractDataTypePtr ds, const std::string &member)
    : ADTMember(new ADTMemberNode(ds, member)) {
}

AbstractDataTypePtr ADTMember::getDS() const {
    return getNode(*this)->ds;
}

std::string ADTMember::getMember() const {
    return getNode(*this)->member;
}

std::ostream &operator<<(std::ostream &os, const Expr &e) {
    Printer p{os};
    p.visit(e);
    return os;
}

bool isConstExpr(Expr e) {
    bool is_const_expr = true;
    match(e, std::function<void(const VariableNode *)>(
                 [&](const VariableNode *op) {
                     if (!op->const_expr) {
                         is_const_expr = false;
                     }
                 }));
    return is_const_expr;
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

Stmt Stmt::whereStmt(Constraint constraint) const {
    auto stmtVars = getVariables(*this);
    auto constraintVars = getVariables(constraint);
    if (!std::includes(stmtVars.begin(), stmtVars.end(), constraintVars.begin(),
                       constraintVars.end(), std::less<Variable>())) {
        throw error::UserError("Putting constraints on variables that are not "
                               "present in statement's scope");
    }
    return Stmt(ptr, constraint);
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
              ctx->match(op->start.getA());
              ctx->match(op->body);
          }),
          std::function<void(const ComputesForNode *op, Matcher *ctx)>([&](const ComputesForNode *op,
                                                                           Matcher *ctx) {
              ctx->match(op->start.getA());
              ctx->match(op->body);
          }),
          std::function<void(const VariableNode *op, Matcher *ctx)>([&](const VariableNode *op,
                                                                        Matcher *) {
              vars.insert(op);
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

std::map<ADTMember, std::tuple<Variable, Expr, Variable>> Stmt::getTileableFields() const {
    std::map<ADTMember, std::tuple<Variable, Expr, Variable>> tileable;
    match(*this,
          std::function<void(const ComputesForNode *op, Matcher *ctx)>([&](const ComputesForNode *op,
                                                                           Matcher *ctx) {
              tileable[op->end] = std::make_tuple(to<Variable>(op->start.getA()), op->start.getB(), op->step);
              ctx->match(op->body);
          }));
    return tileable;
}

std::map<ADTMember, std::tuple<Variable, Expr, Variable>> Stmt::getReducableFields() const {
    std::map<ADTMember, std::tuple<Variable, Expr, Variable>> tileable;
    match(*this,
          std::function<void(const ConsumesForNode *op, Matcher *ctx)>([&](const ConsumesForNode *op,
                                                                           Matcher *ctx) {
              tileable[op->end] = std::make_tuple(to<Variable>(op->start.getA()), op->start.getB(), op->step);
              ctx->match(op->body);
          }));
    return tileable;
}

#define DEFINE_WHERE_METHOD(Type)            \
    Type Type::where(Constraint c) {         \
        return to<Type>(this->whereStmt(c)); \
    }

DEFINE_WHERE_METHOD(Consumes)
DEFINE_WHERE_METHOD(SubsetObj)
DEFINE_WHERE_METHOD(Produces)
DEFINE_WHERE_METHOD(ConsumeMany)
DEFINE_WHERE_METHOD(SubsetObjMany)
DEFINE_WHERE_METHOD(Allocates)
DEFINE_WHERE_METHOD(Pattern)
DEFINE_WHERE_METHOD(Computes)

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

        void visit(const ADTMemberNode *op) {
            if (rw_ds.contains(op->ds)) {
                expr = ADTMember(rw_ds.at(op->ds), op->member);
            } else {
                expr = op;
            }
        }
        void visit(const SubsetNode *op) {
            // Rewrite all the fields.
            std::vector<Expr> rw_expr;
            for (size_t i = 0; i < op->mdFields.size(); i++) {
                rw_expr.push_back(this->rewrite(op->mdFields[i]));
            }
            // Construct the new subset object.
            if (rw_ds.contains(op->data)) {
                stmt = SubsetObj(rw_ds[op->data], rw_expr);
            } else {
                stmt = SubsetObj(op->data, rw_expr);
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
    : Stmt(new const SubsetNode(data, mdFields)) {
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

ConsumeMany Reduce(Assign start, ADTMember end, Variable step, ConsumeMany body,
                   bool parallel) {
    return ConsumeMany(
        new const ConsumesForNode(start, end, step, body, parallel));
}

ConsumeMany Reduce(Assign start, ADTMember end, Variable step, std::vector<SubsetObj> body,
                   bool parallel) {
    return Reduce(start, end, step, SubsetObjMany(body), parallel);
}

ConsumeMany Reduce(Assign start, ADTMember end, Variable step, SubsetObj body,
                   bool parallel) {
    return Reduce(start, end, step, std::vector<SubsetObj>{body}, parallel);
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

Annotation Pattern::occupies(Grid::Unit unit) const {
    return Annotation(*this, unit);
}

Pattern Pattern::refreshVariables() const {
    std::set<Variable> old_vars = getVariables(*this);
    // Generate fresh names for all old variables, except the
    std::map<Variable, Variable> fresh_names;
    for (const auto &v : old_vars) {
        // Otherwise, generate a new name.
        fresh_names[v] = getUniqueName("_gern_" + v.getName());
    }
    Pattern rw_annotation = to<Pattern>(this->replaceVariables(fresh_names));
    return rw_annotation;
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

Annotation::Annotation(const AnnotationNode *n)
    : Stmt(n) {
}

Annotation::Annotation(Pattern p, Grid::Unit unit)
    : Annotation(new const AnnotationNode(p, unit)) {
}

// Annotation::Annotation(Pattern p)
//     : Annotation(p, Grid::Unit::NULL_UNIT) {
// }

Annotation annotate(Pattern p) {
    return Annotation(p, Grid::Unit::NULL_UNIT);
}

Annotation resetUnit(Annotation annot, Grid::Unit unit) {
    return Annotation(annot.getPattern(), unit);
}

Pattern Annotation::getPattern() const {
    return getNode(*this)->p;
}

Grid::Unit Annotation::getOccupiedUnit() const {
    return getNode(*this)->unit;
}

Pattern For(Assign start, ADTMember end, Variable step, Pattern body,
            bool parallel) {
    return Pattern(
        new const ComputesForNode(start, end, step, body, parallel));
}

Pattern For(Assign start, ADTMember end, Variable step,
            Produces produces, Consumes consumes,
            bool parallel) {
    return For(start, end, step, Computes(produces, consumes), parallel);
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
    return ADTMember(*this, member);
}

std::string AbstractDataTypePtr::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

}  // namespace gern