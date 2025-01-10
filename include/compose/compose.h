#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"
#include <vector>

namespace gern {

class Pipeline;

class CompositionVisitorStrict;
/**
 * @brief This class tracks the objects that Gern
 *        can compose together in a pipeline. Currently,
 *         this includes a ComputeFunctionCall, and other pipelines.
 *         By expressing the relationship between these Composition
 *         Object at this level, we can make sub-pipelining and breaks
 *         redundant.
 *
 */
class CompositionObject : public util::Manageable<CompositionObject>,
                          public util::Uncopyable {
public:
    CompositionObject() = default;
    virtual ~CompositionObject() = default;
    virtual void accept(CompositionVisitorStrict *) const = 0;
};

/**
 * @brief This class describes composition of functions
 *         that Gern will generate code for.
 *
 */
class Compose : public util::IntrusivePtr<const CompositionObject> {
public:
    Compose()
        : util::IntrusivePtr<const CompositionObject>(nullptr) {
    }
    Compose(const CompositionObject *n)
        : util::IntrusivePtr<const CompositionObject>(n) {
    }

    Compose(Pipeline p);
    Compose(std::vector<Compose> compose);

    void concretize();
    Compose replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacements) const;

    int numFuncs() const;

    void accept(CompositionVisitorStrict *v) const;

private:
    bool device = false;
    bool concrete = false;
};

std::ostream &operator<<(std::ostream &os, const Compose &);

// A function call has a name,
// a set of arguments, a set of
// templated arguments, and a return type.
struct Function {
    std::string name;
    std::vector<Argument> args;
    // Only int64_t args allowed rn.
    std::vector<Variable> template_args;
    // To model an explict return. Currently, no compute function can return.
    Argument output;

    /**
     * @brief Replace the data-structures in the function.
     *
     * @param Data structures to replace with.
     * @return * Function
     */
    Function replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const;
};

class ComputeFunctionCall : public CompositionObject {
public:
    ComputeFunctionCall() = delete;
    ComputeFunctionCall(Function call,
                        Pattern annotation,
                        std::vector<std::string> header)
        : call(call), annotation(annotation), header(header) {
    }

    Function getCall() const {
        return call;
    }

    const std::string &getName() const {
        return call.name;
    }
    const Pattern &getAnnotation() const {
        return annotation;
    }
    const std::vector<Argument> &getArguments() const {
        return call.args;
    }
    const std::vector<Variable> &getTemplateArguments() const {
        return call.template_args;
    }
    /**
     * @brief Returns the data structure that the function computes as output.
     *
     * @return Pointer to the data structure.
     */
    AbstractDataTypePtr getOutput() const;

    /**
     * @brief Returns the data structures that the function treats as inputs.
     *
     * @return std::set<AbstractDataTypePtr>=
     */
    std::set<AbstractDataTypePtr> getInputs() const;

    /**
     * @brief Returns the name of the header file where the function is
     *        declared.
     *
     */
    std::vector<std::string> getHeader() const {
        return header;
    }

    /**
     * @brief Get the meta data Fields object corresponding to
     *        a data-structure used in the annotation.
     *
     * @param d The data-structure whose fields will be returned.
     * @return std::vector<Expr>
     */
    std::vector<Expr> getMetaDataFields(AbstractDataTypePtr d) const;

    /**
     * @brief Get the Produces meta data fields, this should always be variables.
     *
     * @return std::vector<Variable> The meta-data fields.
     */
    std::vector<Variable> getProducesFields() const;

    /**
     * @brief This function is used to bind a symbolic variable in the
     *        annotation with a user-provided variable. This is useful for
     *        passing in arguments for the function pointer, otherwise,
     *        users do not have a hook for passing arguments properly,
     *        since the argument order is not pre-determined.
     *
     * @param binding  Map that contains the name of the variable in the annotation
     *                 and a gern variable that will be bound to the variable.
     * @return ComputeFunctionCall
     */
    const ComputeFunctionCall *withSymbolic(const std::map<std::string, Variable> &binding);

    /**
     * @brief This function checks whether a passed in variable is a template arg
     *        for a given function.
     *
     * @param v Var to check.
     * @return true
     * @return false
     */
    bool isTemplateArg(Variable v) const;

    ComputeFunctionCall replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const;

    void accept(CompositionVisitorStrict *v) const;

private:
    Function call;
    Pattern annotation;
    std::vector<std::string> header;
};

std::ostream &operator<<(std::ostream &os, const ComputeFunctionCall &f);

template<typename E>
inline bool isa(const CompositionObject *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const CompositionObject *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

}  // namespace gern