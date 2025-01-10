#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/data_dependency_language.h"
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

private:
    AbstractDataTypePtr dataStruct;
};

class VarArg : public ArgumentNode {
public:
    VarArg(Variable v)
        : v(v) {
    }
    Variable getVar() const {
        return v;
    }
    virtual void accept(ArgumentVisitorStrict *) const;

private:
    Variable v;
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
        std::cout << "here ----- " << std::endl;
    }
    Argument(Variable v)
        : Argument(new const VarArg(v)) {
    }
    void accept(ArgumentVisitorStrict *v) const;
};

std::ostream &operator<<(std::ostream &os, const Argument &);
/*
To add arguments to the vector of arguments in the
overload of () for the AbstractFunction class,
an addArgument must be defined for that type of
argument.
*/
[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         AbstractDataTypePtr dataStruct) {
    vector.push_back(Argument(new const DSArg(dataStruct)));
}

[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         Variable v) {
    vector.push_back(Argument(new const VarArg(v)));
}

[[maybe_unused]] static void addArgument(std::vector<Argument> &vector,
                                         Argument a) {
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
inline const E *to(Argument a) {
    return to<E>(a.ptr);
}

}  // namespace gern