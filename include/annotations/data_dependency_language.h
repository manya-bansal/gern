#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/grid.h"
#include "utils/uncopyable.h"
#include <cassert>
#include <map>
#include <memory>
#include <set>

namespace gern {

struct VariableNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;
struct ModNode;
struct EqNode;
struct NeqNode;
struct LeqNode;
struct GeqNode;
struct LessNode;
struct GreaterNode;
struct AndNode;
struct OrNode;
struct AssignNode;

class Expr : public util::IntrusivePtr<const ExprNode> {
public:
    Expr()
        : util::IntrusivePtr<const ExprNode>(nullptr) {
    }
    Expr(const ExprNode *n)
        : util::IntrusivePtr<const ExprNode>(n) {
    }

    Expr(uint8_t);
    Expr(uint16_t);
    Expr(uint32_t);
    Expr(uint64_t);
    Expr(int8_t);
    Expr(int16_t);
    Expr(int32_t);
    Expr(int64_t);
    Expr(float);
    Expr(double);

    bool operator()(const Expr &e) {
        return ptr < e.ptr;
    }

    void accept(ExprVisitorStrict *v) const;
};
std::ostream &operator<<(std::ostream &os, const Expr &);

class Constraint : public util::IntrusivePtr<const ConstraintNode> {
public:
    Constraint()
        : util::IntrusivePtr<const ConstraintNode>(nullptr) {
    }
    Constraint(const ConstraintNode *n)
        : util::IntrusivePtr<const ConstraintNode>(n) {
    }

    virtual Expr getA() const {
        return Expr();
    }

    virtual Expr getB() const {
        return Expr();
    }

    void accept(ConstraintVisitorStrict *v) const;
};

std::ostream &operator<<(std::ostream &os, const Constraint &);

#define DEFINE_BINARY_CLASS(NAME, NODE)    \
    class NAME : public NODE {             \
    public:                                \
        explicit NAME(const NAME##Node *); \
        NAME(Expr a, Expr b);              \
        Expr getA() const;                 \
        Expr getB() const;                 \
        typedef NAME##Node Node;           \
    };

DEFINE_BINARY_CLASS(Add, Expr)
DEFINE_BINARY_CLASS(Sub, Expr)
DEFINE_BINARY_CLASS(Div, Expr)
DEFINE_BINARY_CLASS(Mul, Expr)
DEFINE_BINARY_CLASS(Mod, Expr)

DEFINE_BINARY_CLASS(And, Constraint)
DEFINE_BINARY_CLASS(Or, Constraint)

DEFINE_BINARY_CLASS(Eq, Constraint)
DEFINE_BINARY_CLASS(Neq, Constraint)
DEFINE_BINARY_CLASS(Leq, Constraint)
DEFINE_BINARY_CLASS(Geq, Constraint)
DEFINE_BINARY_CLASS(Less, Constraint)
DEFINE_BINARY_CLASS(Greater, Constraint)

Add operator+(const Expr &, const Expr &);
Sub operator-(const Expr &, const Expr &);
Mul operator*(const Expr &, const Expr &);
Div operator/(const Expr &, const Expr &);
Mod operator%(const Expr &, const Expr &);
Eq operator==(const Expr &, const Expr &);
Neq operator!=(const Expr &, const Expr &);
Leq operator<=(const Expr &, const Expr &);
Geq operator>=(const Expr &, const Expr &);
Less operator<(const Expr &, const Expr &);
Greater operator>(const Expr &, const Expr &);
And operator&&(const Expr &, const Expr &);
Or operator||(const Expr &, const Expr &);

class Assign;

// All variables are current ints.
class Variable : public Expr {
public:
    Variable() = default;
    Variable(const std::string &name);
    Variable(const VariableNode *);

    /**
     *  @brief This function indicates that the
     *          value of the variable is derived
     *          from a grid property. (blockIDx,
     *          etc)
     *
     *  @param p The grid property to bind this variable
     *           to.
     */
    Variable bindToGrid(const Grid::Property &p) const;
    bool isBoundToGrid() const;
    Grid::Property getBoundProperty() const;

    std::string getName() const;
    Datatype getType() const;

    Assign operator=(const Expr &);
    Assign operator+=(const Expr &);

    typedef VariableNode Node;
};

}  // namespace gern

// Defining an std::less overload so that
// std::map<Variable, ....> in Stmt doesn't need to
// take in a special struct. The < operator in Expr
// is already overloaded, so it's not possible to use
// the usual std::less definition.
namespace std {
template<>
struct less<gern::Variable> {
    bool operator()(const gern::Variable &a, const gern::Variable &b) const {
        return a.ptr < b.ptr;
    }
};
}  // namespace std

namespace gern {

struct SubsetNode;
struct SubsetObjManyNode;
struct ProducesNode;
struct ConsumesNode;
struct ConsumesForNode;
struct AllocatesNode;
struct ComputesForNode;
struct ComputesNode;
struct PatternNode;

class Stmt : public util::IntrusivePtr<const StmtNode> {
public:
    Stmt()
        : util::IntrusivePtr<const StmtNode>(nullptr) {
    }
    Stmt(const StmtNode *n)
        : util::IntrusivePtr<const StmtNode>(n) {
    }

    /**
     * @brief Add a constraint to a statement
     *
     *  The function checks that only variables that are in
     *  scope are used within the constraint.
     *
     * @param constraint Constraint to add.
     * @return Stmt New statement with the constraint attached.
     */
    Stmt whereStmt(Constraint constraint) const;
    Constraint getConstraint() const {
        return c;
    }

    std::set<Variable> getDefinedVariables() const;
    std::set<Variable> getIntervalVariables() const;
    Stmt replaceVariables(std::map<Variable, Variable> rw_vars) const;
    Stmt replaceDSArgs(std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds) const;
    void accept(StmtVisitorStrict *v) const;

private:
    Stmt(const StmtNode *n, Constraint c)
        : util::IntrusivePtr<const StmtNode>(n), c(c) {
    }
    Constraint c;
};

std::ostream &operator<<(std::ostream &os, const Stmt &);

template<typename E, typename T>
inline bool isa(const T &e) {
    return e.ptr != nullptr && dynamic_cast<const typename E::Node *>(e.ptr) != nullptr;
}

template<typename E, typename T>
inline const E to(const T &e) {
    assert(isa<E>(e));
    return E(static_cast<const typename E::Node *>(e.ptr));
}

DEFINE_BINARY_CLASS(Assign, Stmt)

class SubsetObj : public Stmt {
public:
    SubsetObj() = default;
    explicit SubsetObj(const SubsetNode *);
    SubsetObj(AbstractDataTypePtr data,
              std::vector<Expr> mdFields);
    std::vector<Expr> getFields();
    SubsetObj where(Constraint);
    AbstractDataTypePtr getDS() const;
    typedef SubsetNode Node;
};

class Produces : public Stmt {
public:
    explicit Produces(const ProducesNode *);
    // Factory method to produce make a produces node.
    static Produces Subset(AbstractDataTypePtr, std::vector<Variable>);
    SubsetObj getSubset() const;
    Produces where(Constraint);
    std::vector<Variable> getFieldsAsVars() const;
    typedef ProducesNode Node;
};

struct ConsumesNode;
class ConsumeMany;

class Consumes : public Stmt {
public:
    explicit Consumes(const ConsumesNode *);
    // Factory method to produce make a consumes node.
    static Consumes Subset(AbstractDataTypePtr, std::vector<Expr>);
    static Consumes Subsets(ConsumeMany);
    Consumes(SubsetObj s);
    Consumes where(Constraint);
    typedef ConsumesNode Node;
};

class ConsumeMany : public Consumes {
public:
    ConsumeMany(const ConsumesNode *s)
        : Consumes(s) {};
    ConsumeMany where(Constraint);
};

class SubsetObjMany : public ConsumeMany {
public:
    SubsetObjMany(const SubsetObjManyNode *);
    SubsetObjMany(const std::vector<SubsetObj> &subsets);
    SubsetObjMany(SubsetObj s)
        : SubsetObjMany(std::vector<SubsetObj>{s}) {
    }
    SubsetObjMany where(Constraint);
    typedef SubsetObjManyNode Node;
};

// This ensures that a consumes node will only ever contain a for loop
// or a list of subsets. In this way, we can leverage the cpp type checker to
// ensures that only legal patterns are written down.
ConsumeMany For(Assign start, Expr end, Expr step, ConsumeMany body,
                bool parallel = false);
ConsumeMany For(Assign start, Expr end, Expr step, std::vector<SubsetObj> body,
                bool parallel = false);
ConsumeMany For(Assign start, Expr end, Expr step, SubsetObj body,
                bool parallel = false);

class Allocates : public Stmt {
public:
    Allocates()
        : Stmt() {
    }
    explicit Allocates(const AllocatesNode *);
    Allocates(Expr reg, Expr smem = Expr());
    Allocates where(Constraint);
    typedef AllocatesNode Node;
};

struct PatternNode;
class Pattern : public Stmt {
public:
    explicit Pattern(const PatternNode *);
    Pattern where(Constraint);
    typedef PatternNode Node;
};

class Computes : public Pattern {
public:
    explicit Computes(const ComputesNode *);
    Computes(Produces p, Consumes c, Allocates a = Allocates());
    Computes where(Constraint);
    typedef ComputesNode Node;
};

// This ensures that a computes node will only ever contain a for loop
// or a (Produces, Consumes) node. In this way, we can leverage the cpp type
// checker to ensures that only legal patterns are written down.
Pattern For(Assign start, Expr end, Expr step, Pattern body,
            bool parallel = false);
// Function so that users do need an explicit compute initialization.
Pattern For(Assign start, Expr end, Expr step,
            Produces produces, Consumes consumes,
            bool parallel = false);

}  // namespace gern