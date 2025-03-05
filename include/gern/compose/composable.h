#pragma once

#include "annotations/argument.h"
#include "annotations/data_dependency_language.h"
#include "annotations/shared_memory_manager.h"
#include "compose/composable_node.h"
#include "utils/uncopyable.h"

namespace gern {

class ComposableVisitorStrict;
class ComputeFunctionCall;

/**
 * @brief Composable manages the lifetime of the ComposableNode object.
 *
 */
class Composable : public util::IntrusivePtr<const ComposableNode> {
public:
    Composable()
        : util::IntrusivePtr<const ComposableNode>(nullptr) {
    }

    Composable(const ComposableNode *n);
    Composable(std::vector<Composable> composed);
    Annotation getAnnotation() const;
    void accept(ComposableVisitorStrict *v) const;
    bool isDeviceLaunch() const;
};

std::ostream &operator<<(std::ostream &os, const Composable &f);

class GlobalNode : public ComposableNode {
public:
    GlobalNode(Composable program,
               std::map<Grid::Dim, Variable> launch_args,
               grid::SharedMemoryManager smem_manager);
    Annotation getAnnotation() const override;
    void accept(ComposableVisitorStrict *) const override;
    Composable program;
    std::map<Grid::Dim, Variable> launch_args;
    grid::SharedMemoryManager smem_manager;
};

// Wrap a function in a global interface, mostly for a nicety.
Composable Global(Composable program,
                  std::map<Grid::Dim, Variable> launch_args = {},
                  grid::SharedMemoryManager smem_manager = {});

/**
 * @brief Computation contains a vector of composable objects.
 *
 */
class Computation : public ComposableNode {
public:
    Computation(std::vector<Composable> composed);
    Annotation getAnnotation() const override;
    void accept(ComposableVisitorStrict *) const override;

    void check_legal();
    void init_annotation();
    void infer_relationships(AbstractDataTypePtr output,
                             std::vector<Variable> output_fields);

    AbstractDataTypePtr output;
    std::vector<Composable> composed;
    std::vector<Assign> declarations;
    std::map<Variable, Expr> declarations_map;

    std::map<AbstractDataTypePtr, std::set<Composable>> consumer_functions;
    Annotation _annotation;
};

class TiledComputation : public ComposableNode {
public:
    TiledComputation(Expr to_tile,
                     Variable v,
                     Composable body,
                     Grid::Unit unit,
                     bool reduce);

    Annotation getAnnotation() const;
    void accept(ComposableVisitorStrict *) const;

    void init_binding();

    Expr to_tile;  // The field that the user wants to tile.
    Variable v;    // The variable that the user will set to concretize the tile
                   // parameter for the computation.
    Composable tiled;
    Variable captured;
    Variable loop_index;  // New loop index.
    Expr start;
    Expr parameter;
    Variable step;
    Annotation _annotation;
    std::map<Variable, Variable> old_to_new;
    Grid::Unit unit{Grid::Unit::UNDEFINED};  // Tracks whether the grid is mapped over a grid.
    bool reduce = false;
};

// This class only exists for the overload.
struct TileDummy {
    TileDummy(Expr to_tile, Variable v,
              bool reduce)
        : to_tile(to_tile), v(v), reduce(reduce) {
    }

    Composable operator()(Composable c);

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
        return operator()(Composable(new const Computation(to_compose)));
    }

    TileDummy operator||(Grid::Unit p);
    Expr to_tile;
    Variable v;
    bool reduce;
    Grid::Unit unit{Grid::Unit::UNDEFINED};
};

TileDummy Tile(Expr tileable, Variable v);
TileDummy Reduce(Expr tileable, Variable v);

template<typename E>
inline bool isa(const ComposableNode *e) {
    return e != nullptr && dynamic_cast<const E *>(e) != nullptr;
}

template<typename E>
inline const E *to(const ComposableNode *e) {
    assert(isa<E>(e));
    return static_cast<const E *>(e);
}

}  // namespace gern