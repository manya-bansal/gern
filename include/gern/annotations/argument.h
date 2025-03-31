#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <cassert>
#include <iostream>

namespace gern {

class ArgumentVisitorStrict;

class ArgumentNode : public util::Manageable<ArgumentNode>,
                     public util::Uncopyable {
public:
    ArgumentNode() = default;
    virtual ~ArgumentNode() = default;
    virtual void accept(ArgumentVisitorStrict *) const = 0;
    virtual std::string getType() const = 0;
};

class DSArg : public ArgumentNode {
public:
    DSArg(AbstractDataTypePtr dataStruct)
        : dataStruct(dataStruct) {
    }
    AbstractDataTypePtr getADTPtr() const {
        return dataStruct;
    }
    virtual void accept(ArgumentVisitorStrict *) const;
    virtual std::string getType() const;

private:
    AbstractDataTypePtr dataStruct;
};

class ExprArg : public ArgumentNode {
public:
    ExprArg(Expr e)
        : e(e) {
    }
    Expr getExpr() const {
        return e;
    }

    Variable getVar() const {
        assert(isa<Variable>(e));
        return to<Variable>(e);
    }

    virtual void accept(ArgumentVisitorStrict *) const;
    virtual std::string getType() const;

private:
    Expr e;
};

class Argument : public util::IntrusivePtr<const ArgumentNode> {
public:
    Argument()
        : util::IntrusivePtr<const ArgumentNode>(nullptr) {
    }
    explicit Argument(const ArgumentNode *a)
        : util::IntrusivePtr<const ArgumentNode>(a) {
    }
    Argument(AbstractDataTypePtr dataStuct)
        : Argument(new const DSArg(dataStuct)) {
    }
    Argument(Expr e)
        : Argument(new const ExprArg(e)) {
    }
    std::string str() const;
    bool isSameTypeAs(Argument) const;
    std::string getType() const;
    void accept(ArgumentVisitorStrict *v) const;
};

// Class used to limit the types of arguments possible
// to pass in FunctionSignature.
class Parameter : public Argument {
public:
    Parameter()
        : Argument() {
    }
    Parameter(Variable v)
        : Argument(v) {
    }
    explicit Parameter(AbstractDataTypePtr ds)
        : Argument(ds) {
    }
};

std::ostream &operator<<(std::ostream &os, const Argument &);
/*
To add arguments to the vector of arguments in the
overload of () for the AbstractFunction  class,
an addArgument must be defined for that type of
argument.
*/
[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         AbstractDataTypePtr dataStruct) {
    vector.push_back(Argument(dataStruct));
}

[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         Variable v) {
    vector.push_back(Argument(v));
}

[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         Argument a) {
    if (!a.defined()) {
        throw error::UserError("Calling with an empty argument");
    }
    vector.push_back(a);
}

[[maybe_unused]] static void addArguments(std::vector<Argument> &arg) {
    // do nothing
    (void)arg;
}

template<typename T, typename... Next>
void addArguments(std::vector<Argument> &arguments, T first,
                  Next... next) {
    addArgument(arguments, first);
    addArguments(arguments, next...);
}

template<typename E>
inline bool isa(const ArgumentNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const ArgumentNode *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

template<typename E>
inline bool isa(Argument a) {
    return isa<E>(a.ptr);
}

template<typename E>
inline bool isa(Parameter a) {
    return isa<E>(a.ptr);
}

template<typename E>
inline const E *to(Argument a) {
    return to<E>(a.ptr);
}

template<typename E>
inline const E *to(Parameter a) {
    return to<E>(a.ptr);
}

}  // namespace gern