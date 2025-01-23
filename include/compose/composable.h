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
    Composable(std::vector<Composable> composed);
    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;
    Pattern getAnnotation() const;
    void accept(ComposableVisitorStrict *v) const;
};

std::ostream &operator<<(std::ostream &os, const Composable &f);

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
    TiledComputation(ADTMember member,
                     Variable v,
                     Composable body);
    std::set<Variable> getVariableArgs() const;
    std::set<Variable> getTemplateArgs() const;
    Pattern getAnnotation() const;
    void accept(ComposableVisitorStrict *) const;

    void init_binding();

    ADTMember field_to_tile;  // The field that the user wants to tile.
    Variable v;               // The variable that the user will set to concretize the tile
                              // parameter for the computation.
    Composable tiled;
    Variable captured;

    Expr start;
    Expr end;
    Variable step;
    Grid::Property property{Grid::Property::UNDEFINED};  // Tracks whether the grid is mapped over a grid.
};

// This class only exists for the overload.
struct TileDummy {
    TileDummy(ADTMember member, Variable v,
              Grid::Property property = Grid::Property::UNDEFINED)
        : member(member), v(v), property(property) {
    }

    template<typename First, typename Second, typename... ToCompose>
    Composable operator()(First first, Second second, ToCompose... c) {
        // Static assertion to ensure all arguments are of type Compose
        static_assert((std::is_same_v<ToCompose, Composable> && ...),
                      "All arguments must be of type Composable");
        static_assert((std::is_same_v<First, Composable>),
                      "All arguments must be of type Composable");
        static_assert((std::is_same_v<Second, Composable>),
                      "All arguments must be of type Composable");
        std::vector<Composable> to_compose{first, second, c...};
        return Composable(new const TiledComputation(member, v,
                                                     Composable(new const Computation(to_compose))));
        // return new const TiledComputation(member, v,
        //                                   Composable(new const Computation(to_compose)),
        //                                   property);
    }

    TileDummy operator||(Grid::Property p);
    Composable operator()(Composable c);

    ADTMember member;
    Variable v;
    Grid::Property property;
};

TileDummy For(ADTMember member, Variable v);

/**
 * @brief Calls takes 1 or more Compose objects and fuses them together.
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

template<typename E>
inline bool isa(const ComposableNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

}  // namespace gern