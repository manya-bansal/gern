#ifndef GERN_ARGUMENTS_H
#define GERN_ARGUMENTS_H

#include "annotations/abstract_nodes.h"
#include "utils/uncopyable.h"
#include <iostream>

namespace gern {

class ArgumentNode : public util::Manageable<ArgumentNode>,
                     public util::Uncopyable {
public:
    ArgumentNode() = default;
    virtual ~ArgumentNode() = default;
    virtual void print(std::ostream &os) const = 0;
};

class Argument : public util::IntrusivePtr<const ArgumentNode> {
public:
    Argument()
        : util::IntrusivePtr<const ArgumentNode>(nullptr) {
    }
    Argument(const ArgumentNode *a)
        : util::IntrusivePtr<const ArgumentNode>(a) {
    }
    void print(std::ostream &os) const {
        ptr->print(os);
    }
};

class DSArg : public ArgumentNode {
public:
    DSArg(std::shared_ptr<const AbstractDataType> dataStruct)
        : dataStruct(dataStruct) {
    }
    void print(std::ostream &os) const override {
        os << *(dataStruct.get());
    }

private:
    std::shared_ptr<const AbstractDataType> dataStruct;
};

void addArgument(std::vector<Argument> &vector,
                 std::shared_ptr<const AbstractDataType> dataStruct) {
    vector.push_back(new const DSArg(dataStruct));
}

void addArguments(std::vector<Argument> &arg) {
    // do nothing
    (void)arg;
}

template<typename T, typename... Next>
void addArguments(std::vector<Argument> &arguments, T first,
                  Next... next) {
    addArgument(arguments, first);
    addArguments(arguments, next...);
}

}  // namespace gern

#endif