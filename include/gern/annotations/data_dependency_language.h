#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/grid.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace gern {

struct VariableNode;
struct ADTMemberNode;
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
struct GridDimNode;
struct PatternNode;
struct AnnotationNode;
struct AssignNode;
struct FunctionSignature;

class Annotation;

class Expr : public util::IntrusivePtr<const ExprNode> {
public:
    Expr()
        : util::IntrusivePtr<const ExprNode>(nullptr) {
    }
    Expr(const ExprNode *n)
        : util::IntrusivePtr<const ExprNode>(n) {
    }

    Expr(uint32_t);
    Expr(uint64_t);
    Expr(int32_t);
    Expr(int64_t);
    Expr(double);
    Expr(Grid::Dim);

    bool operator()(const Expr &e) {
        return ptr < e.ptr;
    }

    void accept(ExprVisitorStrict *v) const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &os, const Expr &);
/**
 * @brief isConstExpr returns whether an expression is a
 *        constant expression (can be evaluated at program
 *        compile time).
 *
 * @return true
 * @return false
 */
bool isConstExpr(Expr);

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
    std::string str() const;
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
    Variable(const std::string &name, bool const_expr);
    Variable(const VariableNode *);

    /**
     *  @brief  bindToGrid indicates that the
     *          value of the variable is derived
     *          from a grid property. (blockIDx,
     *          etc)
     *
     *  @param p The grid property to bind this variable
     *           to.
     */
    Variable bindToGrid(const Grid::Unit &p) const;
    Variable bindToInt64(int64_t) const;
    bool isBoundToGrid() const;
    bool isConstExpr() const;
    bool isBoundToInt64() const;

    /**
     * @brief Returns whether the variable has been set up the user.
     *        This means that either the variable was bound to the grid or
     *        to an int64_t val.
     *
     * @return true
     * @return false
     */
    bool isBound() const;
    int64_t getInt64Val() const;
    Grid::Unit getBoundUnit() const;

    std::string getName() const;
    Datatype getType() const;
    Assign operator+=(const Expr &) const;
    Assign operator=(const Expr &) const;

    typedef VariableNode Node;
};

class AbstractDataType : public util::Manageable<AbstractDataType>,
                         public util::Uncopyable {
public:
    AbstractDataType() = default;
    virtual ~AbstractDataType() = default;

    AbstractDataType(const std::string &name, const std::string &type)
        : name(name), type(type) {
    }

    virtual std::string getName() const {
        return name;
    }

    virtual std::string getType() const {
        return type;
    }

    virtual std::vector<Variable> getFields() const = 0;
    virtual FunctionSignature getAllocateFunction() const = 0;
    virtual FunctionSignature getFreeFunction() const = 0;
    virtual FunctionSignature getInsertFunction() const = 0;
    virtual FunctionSignature getQueryFunction() const = 0;

    // Tracks whether any of the queries need to be free,
    // or if they are actually returning views.
    virtual bool freeQuery() const {
        return false;
    }
    virtual bool insertQuery() const {
        return false;
    }

    virtual bool freeAlloc() const {
        return true;
    }

private:
    std::string name;
    std::string type;
};

class ADTMember;

class AbstractDataTypePtr : public util::IntrusivePtr<const AbstractDataType> {
public:
    AbstractDataTypePtr()
        : util::IntrusivePtr<const AbstractDataType>(nullptr) {
    }
    explicit AbstractDataTypePtr(const AbstractDataType *n)
        : util::IntrusivePtr<const AbstractDataType>(n) {
    }

    std::string getName() const;
    std::string getType() const;
    FunctionSignature getAllocateFunction() const;
    FunctionSignature getQueryFunction() const;
    FunctionSignature getInsertFunction() const;
    std::vector<Variable> getFields() const;
    bool freeQuery() const;
    bool insertQuery() const;
    bool freeAlloc() const;
    std::string str() const;
    ADTMember operator[](std::string) const;
};

std::ostream &operator<<(std::ostream &os, const AbstractDataTypePtr &ads);

class ADTMember : public Expr {
public:
    ADTMember() = default;
    ADTMember(const ADTMemberNode *);
    ADTMember(AbstractDataTypePtr ds, const std::string &field, bool const_expr);
    AbstractDataTypePtr getDS() const;
    std::string getMember() const;
    typedef ADTMemberNode Node;
};

class GridDim : public Expr {
public:
    GridDim(const GridDimNode *);
    GridDim(const Grid::Dim &);
    Grid::Dim getDim() const;
    typedef GridDimNode Node;
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
        return a.getName() < b.getName();
    }
};

template<>
struct less<gern::Expr> {
    bool operator()(const gern::Expr &a, const gern::Expr &b) const {
        return a.ptr < b.ptr;
    }
};

template<>
struct less<gern::AbstractDataTypePtr> {
    bool operator()(const gern::AbstractDataTypePtr &a, const gern::AbstractDataTypePtr &b) const {
        return a.ptr < b.ptr;
    }
};

template<>
struct less<gern::ADTMember> {
    bool operator()(const gern::ADTMember &a, const gern::ADTMember &b) const {
        if (a.getDS() < b.getDS()) return true;   // Compare primary key
        if (b.getDS() < a.getDS()) return false;  // Compare reverse order
        return a.getMember() < b.getMember();     // Compare secondary key
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

    std::set<Variable> getDefinedVariables() const;
    std::set<Variable> getIntervalVariables() const;
    std::map<Variable, Variable> getConsumesIntervalAndStepVars() const;
    std::map<Variable, Variable> getComputesIntervalAndStepVars() const;
    std::map<ADTMember, std::tuple<Variable, Expr, Variable>> getTileableFields() const;
    std::map<ADTMember, std::tuple<Variable, Expr, Variable>> getReducableFields() const;
    void accept(StmtVisitorStrict *v) const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &os, const Stmt &);

template<typename E, typename T>
inline bool isa(const T &e) {
    return e.ptr != nullptr && dynamic_cast<const typename E::Node *>(e.ptr) != nullptr;
}

template<typename E, typename T>
inline const E to(const T &e) {
    if constexpr (std::is_same_v<E, T>) {
        return e;
    } else {
        assert(isa<E>(e));
        return E(static_cast<const typename E::Node *>(e.ptr));
    }
}

DEFINE_BINARY_CLASS(Assign, Stmt)

class SubsetObj : public Stmt {
public:
    SubsetObj() = default;
    explicit SubsetObj(const SubsetNode *);
    SubsetObj(AbstractDataTypePtr data,
              std::vector<Expr> mdFields);
    std::vector<Expr> getFields() const;
    AbstractDataTypePtr getDS() const;
    typedef SubsetNode Node;
};

class Produces : public Stmt {
public:
    explicit Produces(const ProducesNode *);
    // Factory method to produce make a produces node.
    static Produces Subset(AbstractDataTypePtr, std::vector<Variable>);
    SubsetObj getSubset() const;
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
    typedef ConsumesNode Node;
};

class ConsumeMany : public Consumes {
public:
    ConsumeMany(const ConsumesNode *s)
        : Consumes(s) {};
};

class SubsetObjMany : public ConsumeMany {
public:
    SubsetObjMany(const SubsetObjManyNode *);
    SubsetObjMany(const std::vector<SubsetObj> &subsets);
    SubsetObjMany(SubsetObj s)
        : SubsetObjMany(std::vector<SubsetObj>{s}) {
    }
    typedef SubsetObjManyNode Node;
};

// This ensures that a consumes node will only ever contain a for loop
// or a list of subsets. In this way, we can leverage the cpp type checker to
// ensures that only legal patterns are written down.
ConsumeMany Reduce(Assign start, ADTMember end, Variable step, ConsumeMany body,
                   bool parallel = false);
ConsumeMany Reduce(Assign start, ADTMember end, Variable step, std::vector<SubsetObj> body,
                   bool parallel = false);
ConsumeMany Reduce(Assign start, ADTMember end, Variable step, SubsetObj body,
                   bool parallel = false);

class Allocates : public Stmt {
public:
    Allocates()
        : Stmt() {
    }
    explicit Allocates(const AllocatesNode *);
    Allocates(Expr reg, Expr smem = Expr());
    typedef AllocatesNode Node;
};

class Pattern;
class Annotation : public Stmt {
public:
    Annotation() = default;
    Annotation(const AnnotationNode *);
    Annotation(Pattern, std::set<Grid::Unit>, std::vector<Constraint>);
    Pattern getPattern() const;
    std::vector<Constraint> getConstraints() const;

    Annotation assumes(std::vector<Constraint>) const;  // requires is already used as a keyword :(

    template<typename First, typename... Remaining>
    Annotation assumes(First first, Remaining... remaining) const {
        static_assert(std::is_base_of_v<Constraint, First>,
                      "All arguments must be children of Constraint");
        static_assert((std::is_base_of_v<Constraint, Remaining> && ...),
                      "All arguments must be children of Constraint");
        std::vector<Constraint> constraints{first, remaining...};
        return this->assumes(constraints);
    }

    std::set<Grid::Unit> getOccupiedUnits() const;
    typedef AnnotationNode Node;
};

class Pattern : public Stmt {
public:
    Pattern()
        : Stmt() {
    }
    explicit Pattern(const PatternNode *);
    Annotation occupies(std::set<Grid::Unit>) const;

    Annotation assumes(std::vector<Constraint>) const;
    /**
     * @brief assumes adds constraints to the pattern, and
     *        converts it to an annotation.
     *
     * @return Annotation
     */
    template<typename First, typename... Remaining>
    Annotation assumes(First first, Remaining... remaining) const {
        static_assert(std::is_base_of_v<Constraint, First>,
                      "All arguments must be children of Constraint");
        static_assert((std::is_base_of_v<Constraint, Remaining> && ...),
                      "All arguments must be children of Constraint");
        std::vector<Constraint> constraints{first, remaining...};
        return this->assumes(constraints);
    }

    std::vector<SubsetObj> getInputs() const;
    std::vector<Variable> getProducesField() const;
    std::vector<Expr> getRequirement(AbstractDataTypePtr) const;
    SubsetObj getOutput() const;
    typedef PatternNode Node;
};

class Computes : public Pattern {
public:
    explicit Computes(const ComputesNode *);
    Computes(Produces p, Consumes c, Allocates a = Allocates());
    typedef ComputesNode Node;
};

Annotation annotate(Pattern);
Annotation resetUnit(Annotation, std::set<Grid::Unit>);
// This ensures that a computes node will only ever contain a for loop
// or a (Produces, Consumes) node. In this way, we can leverage the cpp type
// checker to ensures that only legal patterns are written down.
Pattern For(Assign start, ADTMember end, Variable step, Pattern body,
            bool parallel = false);
// FunctionSignature so that users do need an explicit compute initialization.
Pattern For(Assign start, ADTMember end, Variable step,
            Produces produces, Consumes consumes,
            bool parallel = false);
}  // namespace gern