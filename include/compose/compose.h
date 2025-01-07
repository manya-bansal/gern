#pragma once

#include "annotations/arguments.h"
#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"
#include <vector>

namespace gern {

class Pipeline;

class CompositionVisitor;
/**
 * @brief This class tracks the objects that Gern
 *        can compose together in a pipeline. Currently,
 *         this includes a FunctionCall, and other pipelines.
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
    virtual void accept(CompositionVisitor *) const = 0;
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

    int numFuncs() const;

    void accept(CompositionVisitor *v) const;

private:
    bool device = false;
    bool concrete = false;
};

std::ostream &operator<<(std::ostream &os, const Compose &);

class FunctionCall : public CompositionObject {
public:
    FunctionCall() = delete;
    FunctionCall(const std::string &name,
                 Pattern annotation,
                 std::vector<Argument> arguments,
                 std::vector<std::string> header)
        : name(name), annotation(annotation),
          arguments(arguments), header(header) {
    }
    const std::string &getName() const {
        return name;
    }
    const Pattern &getAnnotation() const {
        return annotation;
    }
    const std::vector<Argument> &getArguments() const {
        return arguments;
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
     * @return std::set<AbstractDataTypePtr>
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
     * @brief This function is used to bind a symbolic variable in the
     *        annotation with a user-provided variable. This is useful for
     *        passing in arguments for the function pointer, otherwise,
     *        users do not have a hook for passing arguments properly,
     *        since the argument order is not pre-determined.
     *
     * @param binding  Map that contains the name of the variable in the annotation
     *                 and a gern variable that will be bound to the variable.
     * @return FunctionCall
     */
    const FunctionCall *withSymbolic(const std::map<std::string, Variable> &binding);

    void accept(CompositionVisitor *v) const;

private:
    std::string name;
    Pattern annotation;
    std::vector<Argument> arguments;
    std::vector<std::string> header;
};

std::ostream &operator<<(std::ostream &os, const FunctionCall &f);

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