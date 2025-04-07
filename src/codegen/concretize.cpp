#include "codegen/concretize.h"
#include "annotations/rewriter.h"
#include "annotations/rewriter_helpers.h"

namespace gern {

Concretize::Concretize(Composable program)
    : program(program) {
}

LowerIR Concretize::concretize() {
    auto pattern = program.getAnnotation().getPattern();

    // The output and all inputs are in scope, this is
    // passed by the user.
    auto output_adt = pattern.getOutput().getDS();
    adt_in_scope.insert(output_adt, {});
    current_adt.insert(output_adt, SubsetObj(output_adt, {}));

    for (const auto &input : pattern.getInputs()) {
        auto input_adt = input.getDS();
        adt_in_scope.insert(input_adt, {});
        current_adt.insert(input_adt, SubsetObj(input_adt, {}));
    }

    // Now, visit the program.
    this->visit(program);
    return lowerIR;
}

void Concretize::visit(const Computation *node) {

    // Track all the tileable and reducable fields.
    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        declare_intervals(field.first, get<2>(field.second));
    }
    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        declare_intervals(field.first, get<2>(field.second));
    }

    // Track all the definitions that arise as a consequence of chainig the functions.
    for (const auto &c : node->declarations) {
        define_variable(to<Variable>(c.getA()), c.getB());
    }

    std::vector<LowerIR> lowered;
    SubsetObj output_subset = node->getAnnotation().getPattern().getOutput();
    // Now, prepare the output subset.
    auto [ir, ir_insert] = prepare_for_current_scope(output_subset);
    lowered.push_back(ir);

    // Now visit each composition.
    for (const auto &c : node->composed) {
        auto [ir, ir_insert] = prepare_for_current_scope(c.getAnnotation().getPattern().getOutput());
        lowered.push_back(ir);
        this->visit(c);
        lowered.push_back(lowerIR);
        lowered.push_back(ir_insert);
    }

    lowered.push_back(ir_insert);
    lowerIR = new const BlockNode(lowered);
}

void Concretize::visit(const TiledComputation *node) {
    // Visit the body.
    std::vector<LowerIR> lowered;
    for (const auto &rel : node->old_to_new) {
        define_variable(rel.first, rel.second);
    }

    // The user has set up the relationship between step and the variable they will pass in.
    define_variable(node->step, node->v);
    defined.insert(node->v);

    // Track all the tileable and reducable fields.
    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        if (get<0>(field.second).ptr == (node->loop_index).ptr) {  // Skip if this the loop we are materializing.
            continue;
        }
        declare_intervals(field.first, get<2>(field.second));
    }
    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        if (get<0>(field.second).ptr == (node->loop_index).ptr) {
            continue;
        }
        declare_intervals(field.first, get<2>(field.second));
    }

    SubsetObj output_subset = node->getAnnotation().getPattern().getOutput();
    LowerIR accum_insert = LowerIR(new const BlankNode());
    // If it's a reduce, stage the output before lowering the tiled computation.
    if (node->reduce) {
        auto [ir, ir_insert] = prepare_for_current_scope(output_subset);
        lowered.push_back(ir);
        accum_insert = ir_insert;
    }

    adt_in_scope.scope();
    defined.scope();
    tiled_dimensions.scope();
    current_adt.scope();
    iteration_variables.insert(node->loop_index);
    tiled_dimensions.insert(node->parameter, node->v);

    if (isa<Variable>(node->parameter)) {
        // It must be defined, because the user passed it in.
        defined.insert(to<Variable>(node->parameter));
        all_relationships[to<Variable>(node->parameter)] = node->v;
    }

    if (node->reduce) {
        // If we are reducing, then the output must be in scope, since we previously
        // staged it.
        auto output = output_subset.getDS();
        adt_in_scope.insert(output, adt_in_scope.at(output));
        current_adt.insert(output, current_adt.at(output));
    }

    this->visit(node->tiled);

    adt_in_scope.unscope();
    iteration_variables.erase(node->loop_index);
    defined.unscope();
    tiled_dimensions.unscope();
    current_adt.unscope();

    // Do we have another loop tiling this parameter?
    bool has_parent = tiled_dimensions.contains(node->parameter);
    lowerIR = new const IntervalNode(
        has_parent ? (node->loop_index = Expr(0)) : (node->loop_index = node->start),
        has_parent ? tiled_dimensions.at(node->parameter) : Expr(node->parameter),
        node->v,
        lowerIR,
        node->unit);

    lowered.push_back(lowerIR);

    // Insert the current output if needed.
    lowered.push_back(accum_insert);
    lowerIR = new const BlockNode(lowered);
}

void Concretize::visit(const ComputeFunctionCall *node) {
    std::vector<LowerIR> lowered;
    auto annotation = node->getAnnotation();
    // Emit any asserts that we need.
    for (const auto &constraint : annotation.getConstraints()) {
        lowered.push_back(new const AssertNode(get_base_constraint(constraint, {})));
    }

    // For each input and the output, first prepare it and then proceed.
    auto pattern = annotation.getPattern();
    for (const auto &input : pattern.getInputs()) {
        auto [ir, _] = prepare_for_current_scope(input);
        lowered.push_back(ir);
    }
    auto [ir, ir_insert] = prepare_for_current_scope(pattern.getOutput());
    lowered.push_back(ir);

    // Now, generate the function call.
    FunctionCall new_call = node->getCall();
    new_call.args = constuct_arguments(new_call.args);
    new_call.template_args = constuct_arguments(new_call.template_args);

    lowered.push_back(new const ComputeNode(new_call, node->getHeader(),
                                            current_adt.at(pattern.getOutput().getDS()).getDS()));

    lowered.push_back(ir_insert);
    lowerIR = new const BlockNode(lowered);
}

void Concretize::visit(const GlobalNode *node) {
    std::vector<LowerIR> lowered;
    for (const auto &def : node->launch_args) {
        lowered.push_back(new const GridDeclNode(def.first, def.second));
    }
    lowered.push_back(new const SharedMemoryDeclNode(node->smem_size));

    if (node->smem_manager.isInitialized()) {
        lowered.push_back(new const OpaqueCall(node->smem_manager.getInit(),
                                               node->smem_manager.getHeaders()));
    }
    this->visit(node->program);  // Just visit the program.
    lowered.push_back(lowerIR);
    lowerIR = new const BlockNode(lowered);
}

void Concretize::visit(const StageNode *node) {
    // Track all the definitions.
    for (const auto &rel : node->old_to_new) {
        define_variable(rel.first, rel.second);
    }

    // Track all the tileable and reducable fields.
    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        declare_intervals(field.first, get<2>(field.second));
    }

    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        declare_intervals(field.first, get<2>(field.second));
    }

    // Now, stage the subset.
    std::vector<LowerIR> lowered;
    auto [ir, ir_insert] = prepare_for_current_scope(node->staged_subset);
    lowered.push_back(ir);
    // Visit the body it was staged for.
    this->visit(node->body);
    lowered.push_back(lowerIR);

    // If the output of the body is the same as the staged subset, then we need to insert.
    if (node->staged_subset.getDS() == node->body.getAnnotation().getPattern().getOutput().getDS()) {
        lowered.push_back(ir_insert);
    }
    lowerIR = new const BlockNode(lowered);
}

void Concretize::define_variable(Variable var, Expr expr) {
    if (var.isBound() || defined.contains(var)) {
        return;
    }
    all_relationships[var] = expr;
}

Argument Concretize::get_argument(Expr e) {
    auto vars = getVariables(e);
    // All these variables must be defined passed in by the user.
    for (const auto &var : vars) {
        defined.insert(var);
    }
    return Argument(get_base_expr(e, {}));
}

std::vector<Argument> Concretize::constuct_arguments(std::vector<Argument> args) {
    std::vector<Argument> new_args;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg) &&
            current_adt.contains(to<DSArg>(arg)->getADTPtr())) {
            new_args.push_back(Argument(current_adt.at(to<DSArg>(arg)->getADTPtr()).getDS()));
        } else if (isa<ExprArg>(arg)) {
            new_args.push_back(get_argument(to<ExprArg>(arg)->getExpr()));
        } else {
            throw error::InternalError("Unknown argument type: " + arg.str());
        }
    }
    return new_args;
}

void Concretize::declare_intervals(Expr end, Variable step) {
    auto step_expr = get_base_expr(step, {});

    if (isa<Literal>(step_expr)) {
        define_variable(step, end);
    } else {
        define_variable(step, step_expr);
    }

    // If it's a variable, then it must be user provided, so it is defined.
    if (isa<Variable>(end)) {
        defined.insert(to<Variable>(end));
    }
}

/**
 * @brief This function returns the "base" expression for a given expression. The base
 *        expression constists of variables that have been defined in the program (so,
 *        any user provided variables) and iteration variables in the current scope.
 *
 * @param e The expression to get the base expression for.
 * @param stop_at A set of variables that we should travering the graph, used for early exists.
 * @return Expr The base expression for the given expression.
 */

Expr Concretize::get_base_expr(Expr e,
                               std::set<Variable> stop_at) {

    struct locate_parent : public ExprVisitorStrict {
        locate_parent(Expr child,
                      std::map<Variable, Expr> &all_relationships,
                      std::set<Variable> stop_at,
                      std::set<Variable> iteration_variables,
                      util::ScopedSet<Variable> &defined)
            : child(child), all_relationships(all_relationships),
              stop_at(stop_at), iteration_variables(iteration_variables),
              defined(defined) {
        }

        Expr locate() {
            child.accept(this);
            return parent;
        }

#define DEFINE_BINARY_VISIT(NODETYPE, OP)                                                                \
    void visit(const NODETYPE *e) {                                                                      \
        auto a = locate_parent(e->a, all_relationships, stop_at, iteration_variables, defined).locate(); \
        auto b = locate_parent(e->b, all_relationships, stop_at, iteration_variables, defined).locate(); \
        parent = a OP b;                                                                                 \
    }

        DEFINE_BINARY_VISIT(AddNode, +)
        DEFINE_BINARY_VISIT(SubNode, -)
        DEFINE_BINARY_VISIT(MulNode, *)
        DEFINE_BINARY_VISIT(DivNode, /)
        DEFINE_BINARY_VISIT(ModNode, %)

        void visit(const VariableNode *v) {
            if (all_relationships.contains(v) && !stop_at.contains(v)) {
                if (iteration_variables.contains(v)) {
                    parent = v + locate_parent(all_relationships.at(v), all_relationships, stop_at, iteration_variables, defined).locate();
                } else {
                    parent = locate_parent(all_relationships.at(v), all_relationships, stop_at, iteration_variables, defined).locate();
                }
            } else {
                if (stop_at.contains(v)) {
                    parent = 0;  // stop, nowhere to go.
                } else {         // we must be at the base.
                    if (!defined.contains(v) && !iteration_variables.contains(v) && !v->bound) {
                        parent = 0;
                    } else {
                        parent = v;
                    }
                }
            }
        }

        void visit(const ADTMemberNode *e) {
            parent = e;
        }

        void visit(const LiteralNode *e) {
            parent = e;
        }

        void visit(const GridDimNode *e) {
            parent = e;
        }

        Expr child;
        Expr parent;
        std::map<Variable, Expr> &all_relationships;
        std::set<Variable> stop_at;
        std::set<Variable> iteration_variables;
        util::ScopedSet<Variable> &defined;
    };

    return locate_parent(e, all_relationships, stop_at, iteration_variables, defined).locate();
}

FunctionCall Concretize::constructFunctionCall(FunctionSignature f,
                                               std::vector<Variable> ref_md_fields,
                                               std::vector<Expr> true_md_fields) const {

    if (ref_md_fields.size() != true_md_fields.size()) {
        throw error::InternalError("Incorrect number of fields passed!");
    }

    // Put all the fields in a map.
    std::map<Variable, Expr> mappings;
    for (size_t i = 0; i < ref_md_fields.size(); i++) {
        mappings[ref_md_fields[i]] = true_md_fields[i];
    }
    // Now, set up the args.
    std::vector<Argument> new_args;
    for (auto const &a : f.args) {
        new_args.push_back(Argument(mappings.at(to<ExprArg>(a)->getVar())));
    }
    // set up the templated args.
    std::vector<Argument> template_args;
    for (auto const &a : f.template_args) {
        template_args.push_back(Argument(mappings.at(a)));
    }

    FunctionCall f_new = f.constructCall();
    f_new.args = new_args;
    f_new.template_args = template_args;
    f_new.grid = LaunchArguments();
    f_new.block = LaunchArguments();

    return f_new;
}

std::tuple<LowerIR, LowerIR> Concretize::prepare_for_current_scope(SubsetObj subset) {

    LowerIR ir = LowerIR(new const BlankNode());
    LowerIR ir_insert = LowerIR(new const BlankNode());

    auto adt = subset.getDS();

    if (!adt_in_scope.contains_in_current_scope(adt)) {
        if (adt_in_scope.contains(adt)) {
            // If we do have this data structure in scope, then we need to
            // query it from the current adt.
            auto vars_staged_at = adt_in_scope.at(adt);
            std::vector<Expr> fields_expr;
            for (const auto &field : subset.getFields()) {
                fields_expr.push_back(get_base_expr(field, vars_staged_at));
            }
            // Track the scope we defined the adt in.
            adt_in_scope.insert(adt, iteration_variables);

            // Construct the query call from the parent adt.
            AbstractDataTypePtr parent = current_adt.at(adt).getDS();
            auto call = constructFunctionCall(parent.getQueryFunction(), parent.getFields(), fields_expr);

            // Construct the new adt that we are querying.
            AbstractDataTypePtr queried = DummyDS::make(getUniqueName("_query_" + adt.getName()), "auto", adt);
            call.output = Parameter(queried);
            MethodCall method_call = MethodCall(parent, call);
            current_adt.insert(adt, SubsetObj(queried, fields_expr));
            ir = LowerIR(new const QueryNode(adt, queried, fields_expr, method_call));
            // Do we need to insert?
            if (parent.insertQuery()) {
                auto insert_call = constructFunctionCall(parent.getInsertFunction(), parent.getFields(), fields_expr);
                insert_call.args.push_back(queried);
                ir_insert = LowerIR(new const InsertNode(MethodCall(parent, insert_call)));
            }
        } else {
            std::vector<Expr> fields_expr;
            for (const auto &field : subset.getFields()) {
                // We are allocating, do not stop at any variables.
                fields_expr.push_back(get_base_expr(field, {}));
            }
            // Track the scope we defined the adt in.
            adt_in_scope.insert(adt, iteration_variables);

            // Finally, construct the allocate call.
            auto call = constructFunctionCall(adt.getAllocateFunction(), adt.getFields(), fields_expr);
            call.output = Parameter(adt);
            current_adt.insert(adt, SubsetObj(adt, fields_expr));
            ir = LowerIR(new const AllocateNode(call));
        }
    }

    return std::make_tuple(ir, ir_insert);
}

Constraint Concretize::get_base_constraint(Constraint c,
                                           std::set<Variable> stop_at) {
    Constraint c_new = c;
    match(c,
          std::function<void(const EqNode *op)>([&](const EqNode *op) {
              c_new = Eq(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }),
          std::function<void(const NeqNode *op)>([&](const NeqNode *op) {
              c_new = Neq(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }),
          std::function<void(const LessNode *op)>([&](const LessNode *op) {
              c_new = Less(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }),
          std::function<void(const GreaterNode *op)>([&](const GreaterNode *op) {
              c_new = Greater(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }),
          std::function<void(const LeqNode *op)>([&](const LeqNode *op) {
              c_new = Leq(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }),
          std::function<void(const GeqNode *op)>([&](const GeqNode *op) {
              c_new = Geq(get_base_expr(op->a, stop_at), get_base_expr(op->b, stop_at));
          }));
    return c_new;
}

}  // namespace gern
