#ifndef GERN_COMPOSE_H
#define GERN_COMPOSE_H

#include "annotations/arguments.h"
#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"
#include <vector>

namespace gern {
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
class CompositionObject : public util::Manageable<CompositionObject> {
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
    Compose(std::vector<Compose> compose);
    void concretize();
    bool concretized() const;

    void accept(CompositionVisitor *v) const;

private:
    bool concrete = false;
};

std::ostream &operator<<(std::ostream &os, const Compose &);

class FunctionCall : public CompositionObject {
public:
    FunctionCall() = delete;
    FunctionCall(const std::string &name,
                 Pattern annotation,
                 std::vector<Argument> arguments)
        : name(name), annotation(annotation),
          arguments(arguments) {
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
    std::set<AbstractDataTypePtr> getInput() const;
    void accept(CompositionVisitor *v) const;

private:
    std::string name;
    Pattern annotation;
    std::vector<Argument> arguments;
};

class ComposeVec : public CompositionObject {
public:
    ComposeVec(std::vector<Compose> compose)
        : compose(compose) {
    }
    void accept(CompositionVisitor *v) const;
    std::vector<Compose> compose;
};

std::ostream &operator<<(std::ostream &os, const FunctionCall &f);

}  // namespace gern

#endif