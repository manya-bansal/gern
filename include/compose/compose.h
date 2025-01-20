#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "compose/composable.h"
#include "utils/uncopyable.h"
#include <vector>

namespace gern {

class Pipeline;

class CompositionVisitorStrict;
class ComposableVisitorStrict;
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
    virtual Pattern getAnnotation() const = 0;
    virtual bool isTemplateArg(Variable v) const = 0;
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
    Compose(std::vector<Compose> compose, bool fuse);

    Compose callAtDevice();
    Compose callAtHost();
    bool isDeviceCall();

    Pattern getAnnotation() const;
    /**
     * @brief Returns the data structure that the FunctionSignature computes as output.
     *
     * @return Pointer to the data structure.
     */
    AbstractDataTypePtr getOutput() const;

    /**
     * @brief Returns the data structures that the FunctionSignature treats as inputs.
     *
     * @return std::set<AbstractDataTypePtr>=
     */
    std::set<AbstractDataTypePtr> getInputs() const;
    std::vector<Variable> getProducesFields() const;
    std::vector<Expr> getMetaDataFields(AbstractDataTypePtr d) const;
    std::vector<SubsetObj> getAllConsumesSubsets() const;
    bool isTemplateArg(Variable v) const;
    Compose Tile(ADTMember, Variable);
    std::vector<Assign> getBindings() const;
    Compose setBindings(std::vector<Assign>);

    Compose replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacements) const;
    void accept(CompositionVisitorStrict *v) const;

private:
    std::vector<Assign> bindings;
    bool is_device_call = false;
};

std::ostream &operator<<(std::ostream &os, const Compose &);
Compose For(ADTMember, Variable, Compose);

/**
 * @brief Tiles takes 1 or more Compose objects and fuses them together.
 *
 * @param functions The list of functions to fuse
 * @return Compose
 */
template<typename... ToCompose>
Compose Tile(ToCompose... c) {
    // Static assertion to ensure all arguments are of type Compose
    static_assert((std::is_same_v<ToCompose, Compose> && ...), "All arguments must be of type Compose");
    std::vector<Compose> to_compose{c...};
    return Compose(to_compose, true);
}

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

class ComputeFunctionCall : public CompositionObject,
                            public ComposableNode {
public:
    ComputeFunctionCall() = delete;
    ComputeFunctionCall(FunctionCall call,
                        Pattern annotation,
                        std::vector<std::string> header)
        : call(call), annotation(annotation), header(header) {
    }

    FunctionCall getCall() const {
        return call;
    }

    const std::string &getName() const {
        return call.name;
    }

    Pattern getAnnotation() const {
        return annotation;
    }

    const std::vector<Argument> &getArguments() const {
        return call.args;
    }
    const std::vector<Expr> &getTemplateArguments() const {
        return call.template_args;
    }
    /**
     * @brief Returns the data structure that the FunctionSignature computes as output.
     *
     * @return Pointer to the data structure.
     */
    AbstractDataTypePtr getOutput() const;

    /**
     * @brief Returns the data structures that the FunctionSignature treats as inputs.
     *
     * @return std::set<AbstractDataTypePtr>=
     */
    std::set<AbstractDataTypePtr> getInputs() const;

    /**
     * @brief Returns the name of the header file where the FunctionSignature is
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
     * @brief This FunctionSignature checks whether a passed in variable is a template arg
     *        for a given function.
     *
     * @param v Var to check.
     * @return true
     * @return false
     */
    bool isTemplateArg(Variable v) const;

    ComputeFunctionCall replaceAllDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> replacement) const;
    // Generate new variables for everything except variables passed as argument.
    const ComputeFunctionCall *refreshVariable() const;
    std::set<Variable> getVariableArgs() const;

    void accept(CompositionVisitorStrict *v) const;
    void accept(ComposableVisitorStrict *) const;

private:
    FunctionCall call;
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

template<typename E>
inline bool isa(const Compose c) {
    return isa<E>(c.ptr);
}

template<typename E>
inline const E *to(const Compose c) {
    assert(isa<E>(c));
    return to<E>(c.ptr);
}

}  // namespace gern