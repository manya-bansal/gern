#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"
#include <cassert>
#include <iostream>

namespace gern {

enum ArgumentType {
    UNDEFINED,
    DATA_STRUCTURE,
    GERN_VARIABLE,  // Gern variable currently implies int64_t type;
};

class ArgumentNode : public util::Manageable<ArgumentNode>,
                     public util::Uncopyable {
public:
    ArgumentNode() = default;
    ArgumentNode(ArgumentType arg_type)
        : arg_type(arg_type) {
    }
    virtual ~ArgumentNode() = default;
    virtual void print(std::ostream &os) const = 0;
    ArgumentType getType() const {
        return arg_type;
    }

private:
    ArgumentType arg_type = UNDEFINED;
};

class DSArg : public ArgumentNode {
public:
    DSArg(AbstractDataTypePtr dataStruct)
        : ArgumentNode(DATA_STRUCTURE),
          dataStruct(dataStruct) {
    }
    void print(std::ostream &os) const override {
        os << *(dataStruct.get());
    }
    AbstractDataTypePtr getADTPtr() const {
        return dataStruct;
    }

private:
    AbstractDataTypePtr dataStruct;
};

class VarArg : public ArgumentNode {
public:
    VarArg(Variable v)
        : ArgumentNode(GERN_VARIABLE), v(v) {
    }
    void print(std::ostream &os) const override {
        os << v;
    }
    Variable getVar() const {
        return v;
    }

private:
    Variable v;
};

class Argument : public util::IntrusivePtr<const ArgumentNode> {
public:
    Argument()
        : util::IntrusivePtr<const ArgumentNode>(nullptr) {
    }
    Argument(const ArgumentNode *a)
        : util::IntrusivePtr<const ArgumentNode>(a) {
    }
    Argument(AbstractDataTypePtr dataStuct)
        : Argument(new const DSArg(dataStuct)) {
    }
    Argument(Variable v)
        : Argument(new const VarArg(v)) {
    }
    void print(std::ostream &os) const {
        ptr->print(os);
    }
    ArgumentType getType() const {
        return ptr->getType();
    }
};

/*
To add arguments to the vector of arguments in the
overload of () for the AbstractFunction class,
an addArgument must be defined for that type of
argument.
*/
[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         AbstractDataTypePtr dataStruct) {
    vector.push_back(new const DSArg(dataStruct));
}

[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         Variable v) {
    vector.push_back(new const VarArg(v));
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

}  // namespace gern