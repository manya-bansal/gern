#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "compose/composable.h"
#include "compose/compose.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <any>
#include <string>

namespace gern {

/**
 * @brief To add a FunctionSignature to gern, the user needs to
 *        define an instance of the AbstractFunction  class.
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
    virtual Pattern getAnnotation() = 0;
    virtual std::vector<std::string> getHeader() = 0;
    virtual FunctionSignature getFunction() = 0;

    AbstractFunction &operator[](const std::map<std::string, Variable> &replacements) {
        bindVariables(replacements);
        return *this;
    }

    template<typename T>
    Compose operator()(T argument) {
        std::vector<Argument> arguments;
        addArguments(arguments, argument);
        return generateComputeFunctionCall(arguments);
    }

    template<typename FirstT, typename... Next>
    Compose operator()(FirstT first, Next... remaining) {
        std::vector<Argument> arguments;
        addArguments(arguments, first, remaining...);
        return generateComputeFunctionCall(arguments);
    }

    Compose operator()() {
        if (getFunction().args.size() != 0) {
            throw error::UserError("Called FunctionSignature " + getFunction().name + " with 0 arguments");
        }
        return generateComputeFunctionCall({});
    }

    template<typename FirstT, typename... Next>
    Composable construct(FirstT first, Next... remaining) {
        std::vector<Argument> arguments;
        addArguments(arguments, first, remaining...);
        return constructComposableObject(arguments);
    }

private:
    Compose generateComputeFunctionCall(std::vector<Argument> concrete_arguments);
    Composable constructComposableObject(std::vector<Argument> concrete_arguments);
    /**
     * @brief This FunctionSignature actually performs the binding, and checks
     *        the following conditions:
     *
     *        1. Interval variables can only be bound to ID property.
     *        2. Non-interval variables cannot be bound to ID property.
     *        3. If a variable is already completely determined, then it
     *           cannot be bound.
     *
     */
    void bindVariables(const std::map<std::string, Variable> &);
    std::map<std::string, Variable> bindings;
};

}  // namespace gern