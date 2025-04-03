#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/grid.h"
#include "utils/error.h"
#include "utils/name_generator.h"
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
    Datatype getType() const;
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

    virtual Expr getA() const;
    virtual Expr getB() const;

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
    Variable(const std::string &name,
             Datatype type = Datatype::Int64,
             bool const_expr = false);
    Variable(const VariableNode *);

    Variable bind(int64_t) const;
    bool isConstExpr() const;

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
    Datatype getDatatype() const;

    std::string getName() const;
    Datatype getType() const;
    Assign operator+=(const Expr &) const;
    Assign operator=(const Expr &) const;

    typedef VariableNode Node;
};

class AbstractDataType : public util::Manageable<AbstractDataType> {
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

    explicit AbstractDataTypePtr(const AbstractDataType *ptr)
        : util::IntrusivePtr<const AbstractDataType>(ptr) {
    }

    template<typename T, std::enable_if_t<std::is_base_of_v<AbstractDataType, T>, int> = 0>
    AbstractDataTypePtr(const T obj)
        : util::IntrusivePtr<const AbstractDataType>(new T(obj)) {
    }

    std::string getName() const;
    std::string getType() const;
    FunctionSignature getAllocateFunction() const;
    FunctionSignature getQueryFunction() const;
    FunctionSignature getInsertFunction() const;
    FunctionSignature getFreeFunction() const;
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

}  // namespace gern