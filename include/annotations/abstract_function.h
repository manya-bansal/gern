#ifndef GERN_ABSTRACT_FUNCTION_H
#define GERN_ABSTRACT_FUNCTION_H

#include "annotations/arguments.h"
#include "annotations/data_dependency_language.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <any>
#include <string>

namespace gern {

class FunctionCall {
public:
    FunctionCall() = delete;
    FunctionCall(const std::string &name,
                 Stmt annotation,
                 std::vector<Argument> arguments)
        : name(name), annotation(annotation),
          arguments(arguments) {
    }
    const std::string &getName() const {
        return name;
    }
    const Stmt &getAnnotation() const {
        return annotation;
    }
    const std::vector<Argument> &getArguments() const {
        return arguments;
    }

private:
    std::string name;
    Stmt annotation;
    std::vector<Argument> arguments;
};

std::ostream &operator<<(std::ostream &os, const FunctionCall &f);

/**
 * @brief To add a function to gern, the user needs to
 *        define an instance of the AbstractFunction class.
 *
 * The class consists of  x virtual function;
 *
 */
class AbstractFunction {
public:
    AbstractFunction() = default;
    virtual ~AbstractFunction() = default;
    // Not marking these functions as constant
    // because users can meta-program these class
    // using cpp if they would like.
    virtual std::string getName() = 0;
    virtual Stmt getAnnotation() = 0;
    virtual std::vector<Argument> getArguments() = 0;

    template<typename T>
    FunctionCall operator()(T argument) {
        std::vector<Argument> arguments;
        addArguments(arguments, argument);
        return FunctionCall(getName(), getAnnotation(), arguments);
    }

    template<typename FirstT, typename... Next>
    FunctionCall operator()(FirstT first, Next... remaining) {
        std::vector<Argument> arguments;
        addArguments(arguments, first, remaining...);
        return FunctionCall(getName(), getAnnotation(), arguments);
    }

    FunctionCall operator()() {
        if (getArguments().size() != 0) {
            throw error::UserError("Called function " + getName() + " with 0 arguments");
        }
        return FunctionCall(getName(),
                            getAnnotation(),
                            std::vector<Argument>());
    }

private:
    Stmt rewriteAnnotWithConcreteArgs(std::vector<Argument> concrete_arguments);
};

}  // namespace gern
#endif