#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "compose/composable.h"
#include "utils/uncopyable.h"
#include <vector>

namespace gern {

class ComposableVisitorStrict;

// For making an actual function call.
struct FunctionCall {
    std::string name;
    std::vector<Argument> args;
    std::vector<Expr> template_args;
    Parameter output = Parameter();

    /**
     * @brief Replace the data-structures in this function call.
     *
     * @param Data structures to replace with.
     * @return * Function
     */
    FunctionCall replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const;
};

// A FunctionSignature call has a name,
// a set of arguments, a set of
// templated arguments, and a return type.
struct FunctionSignature {
    std::string name;
    std::vector<Parameter> args = {};
    // Only int64_t args allowed rn.
    std::vector<Variable> template_args = {};
    // To model an explict return. Currently, no compute FunctionSignature can return.
    Parameter output = Parameter();
    bool device = false;
    FunctionCall constructCall() const;
};

std::ostream &operator<<(std::ostream &os, const FunctionSignature &f);
std::ostream &operator<<(std::ostream &os, const FunctionCall &f);

class ComputeFunctionCall : public ComposableNode {
public:
    ComputeFunctionCall() = delete;
    ComputeFunctionCall(FunctionCall call,
                        Annotation annotation,
                        std::vector<std::string> header)
        : call(call), annotation(annotation), header(header) {
    }

    FunctionCall getCall() const {
        return call;
    }

    Annotation getAnnotation() const override {
        return annotation;
    }

    /**
     * @brief Returns the name of the header file where the FunctionSignature is
     *        declared.
     *
     */
    std::vector<std::string> getHeader() const {
        return header;
    }

    // Generate new variables for everything except variables passed as argument.
    const ComputeFunctionCall *refreshVariable() const;
    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;

    void accept(ComposableVisitorStrict *) const;

private:
    FunctionCall call;
    Annotation annotation;
    std::vector<std::string> header;
};

using ComputeFunctionCallPtr = const ComputeFunctionCall *;

std::ostream &operator<<(std::ostream &os, const ComputeFunctionCall &f);

}  // namespace gern