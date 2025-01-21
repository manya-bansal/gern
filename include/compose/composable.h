#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"

namespace gern {

class ComposableVisitorStrict;
class ComputeFunctionCall;

/**
 * @brief ComposableNode is an object that Gern understands
 *        how to compose.
 *
 */
class ComposableNode : public util::Manageable<ComposableNode>,
                       public util::Uncopyable {
public:
    ComposableNode() = default;
    virtual ~ComposableNode() = default;
    virtual void accept(ComposableVisitorStrict *) const = 0;
    virtual std::set<Variable> getVariableArgs() const = 0;
    virtual std::set<Variable> getTemplateArgs() const = 0;
    virtual Pattern getAnnotation() const = 0;
};

/**
 * @brief Composable manages the lifetime of the ComposableNode object.
 *
 */
class Composable : public util::IntrusivePtr<const ComposableNode> {
public:
    Composable()
        : util::IntrusivePtr<const ComposableNode>(nullptr) {
    }
    Composable(const ComposableNode *n)
        : util::IntrusivePtr<const ComposableNode>(n) {
    }

    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;
    Pattern getAnnotation() const;
    void accept(ComposableVisitorStrict *v) const;
};

/**
 * @brief Computation contains a vector of composable objects.
 *
 */
class Computation : public ComposableNode {
public:
    Computation(std::vector<Composable> composed);

    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;

    Pattern getAnnotation() const;
    void accept(ComposableVisitorStrict *) const;

    void check_legal();
    void init_args();
    void init_annotation();
    void init_template_args();
    void infer_relationships(AbstractDataTypePtr output,
                             std::vector<Variable> output_fields);

    AbstractDataTypePtr output;
    std::vector<Composable> composed;
    std::vector<Assign> declarations;
    std::set<Variable> variable_args;
    std::set<Variable> template_args;
    std::map<AbstractDataTypePtr, std::set<Composable>> consumer_functions;
    Pattern _annotation;
};

class TiledComputation : public ComposableNode {
public:
    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;

    void accept(ComposableVisitorStrict *) const;
    Pattern getAnnotation() const;

    ADTMember field_to_tile;  // The field that the user wants to tile.
    Variable v;               // The variable that the user will set to concretize the tile
                              // parameter for the computation.
    Composable tiled;
};

/**
 * @brief Tiles takes 1 or more Compose objects and fuses them together.
 *
 * @param functions The list of functions to fuse
 * @return Compose
 */
template<typename... ToCompose>
Composable Call(ToCompose... c) {
    // Static assertion to ensure all arguments are of type Compose
    static_assert((std::is_same_v<ToCompose, Composable> && ...),
                  "All arguments must be of type Composable");
    std::vector<Composable> to_compose{c...};
    return new const Computation(to_compose);
}

}  // namespace gern