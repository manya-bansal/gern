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
    void accept(ComposableVisitorStrict *) const;

    void init_args();
    std::vector<Composable> composed;
    std::set<Variable> variable_args;
};

class TiledComputation : public ComposableNode {
public:
    std::set<Variable> getVariableArgs() const;
    void accept(ComposableVisitorStrict *) const;
    ADTMember field_to_tile;  // The field that the user wants to tile.
    Variable v;               // The variable that the user will set to concretize the tile
                              // parameter for the computation.
    Composable tiled;
};

}  // namespace gern