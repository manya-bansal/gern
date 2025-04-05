#pragma once

#include "annotations/abstract_function.h"
#include "codegen/lower.h"
#include "compose/composable.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"
#include "utils/scoped_map.h"
#include "utils/scoped_set.h"
#include "utils/uncopyable.h"

namespace gern {

class Concretize : public ComposableVisitorStrict {
public:
    Concretize(Composable program);

    LowerIR concretize();

private:
    Composable program;
    Composable concrete_program;
    LowerIR lowerIR;

    util::ScopedMap<AbstractDataTypePtr, std::set<Variable>> adt_in_scope;

    using ComposableVisitorStrict::visit;
    void visit(const Computation *);
    void visit(const TiledComputation *);
    void visit(const ComputeFunctionCall *);
    void visit(const GlobalNode *);
    void visit(const StageNode *);

    void define_variable(Variable var, Expr expr);

    std::map<Variable, Expr> all_relationships;
    util::ScopedSet<Variable> defined;
    std::set<Variable> iteration_variables;
    util::ScopedMap<AbstractDataTypePtr, SubsetObj> current_adt;
    util::ScopedMap<Expr, Variable> tiled_dimensions;

    LowerIR declare_intervals(Variable i, Expr start, Expr end, Variable step);
    LowerIR generate_definition(Assign assign, bool check_const_expr = true);

    LowerIR prepare_for_current_scope(SubsetObj subset);
    Expr get_base_expr(Expr e,
                       std::map<Variable, Expr> &all_relationships,
                       std::set<Variable> stop_at);
    FunctionCall constructFunctionCall(FunctionSignature f,
                                       std::vector<Variable> ref_md_fields,
                                       std::vector<Expr> true_md_fields) const;
};

}  // namespace gern