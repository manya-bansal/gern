#ifndef GERN_ABSTRACT_FUNCTION_H
#define GERN_ABSTRACT_FUNCTION_H

#include "annotations/arguments.h"
#include "annotations/data_dependency_language.h"
#include "compose/compose.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <any>
#include <string>

namespace gern {

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
    virtual Pattern getAnnotation() = 0;
    virtual std::vector<Argument> getArguments() = 0;
    virtual std::vector<std::string> getHeader() = 0;

    AbstractFunction &operator[](const std::map<std::string, Variable> &replacements) {
        bindVariables(replacements);
        return *this;
    }

    template<typename T>
    const FunctionCall *operator()(T argument) {
        std::vector<Argument> arguments;
        addArguments(arguments, argument);
        return new const FunctionCall(getName(),
                                      rewriteAnnotWithConcreteArgs(arguments),
                                      arguments, getHeader());
    }

    template<typename FirstT, typename... Next>
    const FunctionCall *operator()(FirstT first, Next... remaining) {
        std::vector<Argument> arguments;
        addArguments(arguments, first, remaining...);
        return new const FunctionCall(getName(),
                                      rewriteAnnotWithConcreteArgs(arguments),
                                      arguments, getHeader());
    }

    const FunctionCall *operator()() {
        if (getArguments().size() != 0) {
            throw error::UserError("Called function " + getName() + " with 0 arguments");
        }
        return new const FunctionCall(getName(),
                                      rewriteAnnotWithConcreteArgs({}),
                                      std::vector<Argument>(), getHeader());
    }

private:
    Pattern rewriteAnnotWithConcreteArgs(std::vector<Argument> concrete_arguments);
    /**
     * @brief This function actually performs the binding, and checks
     *        the following conditions:
     *
     *        1. Interval variables can only be bound to ID property.
     *        2. Non-interval varaibles cannot be bound to ID property.
     *        3. If a variable is already completely determined, then it
     *           cannot be bound.
     *
     */
    void bindVariables(const std::map<std::string, Variable> &);
    std::map<std::string, Variable> bindings;
};

}  // namespace gern
#endif