#include "codegen/concretize.h"
#include "annotations/rewriter.h"
#include "annotations/rewriter_helpers.h"

namespace gern {

Concretize::Concretize(Composable program)
    : program(program) {
    std::cout << program << std::endl;
}

LowerIR Concretize::concretize() {
    // The output and all inputs are in scope, this is
    // passed by the user.
    auto pattern = program.getAnnotation().getPattern();
    adt_in_scope.insert(pattern.getOutput().getDS(), {});
    current_adt.insert(pattern.getOutput().getDS(), pattern.getOutput().getDS());

    for (const auto &input : pattern.getInputs()) {
        adt_in_scope.insert(input.getDS(), {});
        current_adt.insert(input.getDS(), input.getDS());
    }

    this->visit(program);
    std::cout << lowerIR << std::endl;
    return lowerIR;
}

Expr Concretize::get_base_expr(Expr e,
                               std::map<Variable, Expr> &all_relationships,
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
            // std::cout << "Visiting variable: " << v->name << std::endl;
            // for (const auto &rel : all_relationships) {
            //     std::cout << rel.first.str() << " -> " << rel.second.str() << std::endl;
            // }
            if (all_relationships.contains(v) && !stop_at.contains(v)) {
                if (iteration_variables.contains(v)) {
                    parent = v + locate_parent(all_relationships.at(v), all_relationships, stop_at, iteration_variables, defined).locate();
                } else {
                    parent = locate_parent(all_relationships.at(v), all_relationships, stop_at, iteration_variables, defined).locate();
                }
            } else {
                if (stop_at.contains(v) || (!defined.contains(v) && !iteration_variables.contains(v))) {
                    parent = 0;  // stop, nowhere to go.
                } else {         // we must be at the base.
                    parent = v;
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

static void printRelationships(Variable v,
                               util::ScopedMap<Variable, Expr> &all_relationships,
                               std::set<Variable> stop_at) {
    if (stop_at.count(v)) {
        return;
    }

    if (all_relationships.contains(v)) {
        std::cout << v.str() << " -> " << all_relationships.at(v).str() << std::endl;
        auto vars = getVariables(all_relationships.at(v));
        for (const auto &var : vars) {
            printRelationships(var, all_relationships, stop_at);
        }
    }
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
    std::vector<Expr> template_args;
    for (auto const &a : f.template_args) {
        template_args.push_back(mappings.at(a));
    }

    FunctionCall f_new = f.constructCall();
    f_new.args = new_args;
    f_new.template_args = template_args;
    f_new.grid = LaunchArguments();
    f_new.block = LaunchArguments();

    return f_new;
}

LowerIR Concretize::prepare_for_current_scope(SubsetObj subset) {

    LowerIR ir = LowerIR(new const BlankNode());

    auto adt = subset.getDS();
    auto fields = getVariables(subset);

    if (!adt_in_scope.contains_in_current_scope(adt)) {
        if (adt_in_scope.contains(adt)) {
            // If we do have this data structure in scope, then we need to
            // query it.
            auto staged_vars = adt_in_scope.at(adt);
            std::vector<Expr> fields_expr;
            for (const auto &field : subset.getFields()) {
                fields_expr.push_back(get_base_expr(field, all_relationships, staged_vars));
            }

            // Insert for current iteration variables.
            adt_in_scope.insert(adt, iteration_variables);

            AbstractDataTypePtr parent = current_adt.at(adt);
            std::cout << "Current ADT: " << current_adt << std::endl;
            auto call = constructFunctionCall(parent.getQueryFunction(), parent.getFields(), fields_expr);

            AbstractDataTypePtr queried = DummyDS::make(getUniqueName("_query_" + adt.getName()), "auto", adt);
            call.output = Parameter(queried);
            MethodCall method_call = MethodCall(parent, call);
            current_adt.insert(adt, queried);
            ir = LowerIR(new const QueryNode(adt, queried, fields_expr, method_call));

        } else {

            std::vector<Expr> fields_expr;
            for (const auto &field : subset.getFields()) {
                fields_expr.push_back(get_base_expr(field, all_relationships, {}));
            }

            adt_in_scope.insert(adt, iteration_variables);

            auto call = constructFunctionCall(adt.getAllocateFunction(), adt.getFields(), fields_expr);
            call.output = Parameter(adt);
            current_adt.insert(adt, adt);
            ir = LowerIR(new const AllocateNode(call));
        }
    }

    return ir;
}

LowerIR Concretize::generate_definition(Assign assign, bool check_const_expr) {
    Variable var = to<Variable>(assign.getA());
    all_relationships[var] = assign.getB();

    defined.insert(var);
    return LowerIR(new DefNode(assign, check_const_expr && isConstExpr(assign.getB())));
}

LowerIR Concretize::declare_intervals(Variable i, Expr start, Expr end, Variable step) {
    // TODO: Implement this.
    std::vector<LowerIR> lowered;
    auto expr = get_base_expr(i, all_relationships, {});
    if (isa<Literal>(expr)) {
        // all_relationships[i] = start;
        // lowered.push_back(generate_definition(i = start, false));
    } else {
        // all_relationships[i] = expr;
        // lowered.push_back(generate_definition(i = expr, false));
    }

    auto step_expr = get_base_expr(step, all_relationships, {});

    if (isa<Literal>(step_expr)) {
        all_relationships[step] = end;
        // lowered.push_back(generate_definition(step = end, true));
    } else {
        all_relationships[step] = step_expr;
        // lowered.push_back(generate_definition(step = step_expr, true));
    }

    if (isa<Variable>(end)) {
        // This must be declared.
        defined.insert(to<Variable>(end));
    }
    return LowerIR(new const BlockNode(lowered));
}

void Concretize::visit(const Computation *node) {
    std::vector<LowerIR> lowered;

    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }

    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }
    for (const auto &c : node->declarations) {
        all_relationships[to<Variable>(c.getA())] = c.getB();
        // lowered.push_back(generate_definition(c));
    }
    // Stage the output.
    lowered.push_back(prepare_for_current_scope(node->getAnnotation().getPattern().getOutput()));

    // Now visit.
    for (const auto &c : node->composed) {
        lowered.push_back(prepare_for_current_scope(c.getAnnotation().getPattern().getOutput()));
        this->visit(c);
        lowered.push_back(lowerIR);
    }

    lowerIR = new const BlockNode(lowered);
}

void Concretize::visit(const TiledComputation *node) {
    // Visit the body.
    // all_relationships.scope();
    std::vector<LowerIR> lowered;
    for (const auto &rel : node->old_to_new) {
        all_relationships[rel.first] = rel.second;
    }

    all_relationships[node->step] = node->v;
    defined.insert(node->v);

    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        if (get<0>(field.second).ptr == (node->loop_index).ptr) {
            continue;
        }
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }

    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        if (get<0>(field.second).ptr == (node->loop_index).ptr) {
            continue;
        }
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }

    if (node->reduce) {
        // If we are reducing, then we stage the output, and then proceed.
        lowered.push_back(prepare_for_current_scope(node->tiled.getAnnotation().getPattern().getOutput()));
    }

    adt_in_scope.scope();
    defined.scope();
    tiled_dimensions.scope();
    current_adt.scope();

    iteration_variables.insert(node->loop_index);
    tiled_dimensions.insert(node->parameter, node->v);

    if (node->reduce) {
        // If we are reducing, then the output must be in scope, since we previously
        // staged it.
        auto output = node->tiled.getAnnotation().getPattern().getOutput().getDS();
        auto output_fields = adt_in_scope.at(node->tiled.getAnnotation().getPattern().getOutput().getDS());
        adt_in_scope.insert(output, output_fields);
        current_adt.insert(output, current_adt.at(output));
    }

    this->visit(node->tiled);
    lowered.push_back(lowerIR);

    adt_in_scope.unscope();
    iteration_variables.erase(node->loop_index);
    defined.unscope();
    tiled_dimensions.unscope();
    current_adt.unscope();

    lowerIR = new const BlockNode(lowered);

    auto expr = get_base_expr(node->step, all_relationships, {});
    bool has_parent = tiled_dimensions.contains(node->parameter);

    lowerIR = new const IntervalNode(
        has_parent ? (node->loop_index = Expr(0)) : (node->loop_index = node->start),
        has_parent ? tiled_dimensions.at(node->parameter) : Expr(node->parameter),
        node->v,
        lowerIR,
        node->unit);
}

void Concretize::visit(const ComputeFunctionCall *node) {
    // For each input and output, first stage it and then proceed.
    auto pattern = node->getAnnotation().getPattern();
    std::vector<LowerIR> lowered;
    for (const auto &input : pattern.getInputs()) {
        lowered.push_back(prepare_for_current_scope(input));
    }

    lowered.push_back(prepare_for_current_scope(pattern.getOutput()));

    // Now, generate the function call.
    FunctionCall call = node->getCall();
    std::vector<Argument> new_args;
    auto args = call.args;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg) &&
            current_adt.contains(to<DSArg>(arg)->getADTPtr())) {
            new_args.push_back(Argument(current_adt.at(to<DSArg>(arg)->getADTPtr())));
        } else if (isa<ExprArg>(arg)) {
            // Do we have a value floating around?
            if (tiled_dimensions.contains(to<ExprArg>(arg)->getVar())) {
                new_args.push_back(Argument(tiled_dimensions.at(to<ExprArg>(arg)->getVar())));
            } else {
                new_args.push_back(Argument(to<ExprArg>(arg)->getVar()));
            }
        } else {
            throw error::InternalError("Unknown argument type: " + arg.str());
        }
    }

    std::vector<Expr> new_template_args;
    auto template_args = call.template_args;
    for (const auto &arg : template_args) {
        if (tiled_dimensions.contains(arg)) {
            new_template_args.push_back(tiled_dimensions.at(arg));
        } else {
            new_template_args.push_back(arg);
        }
    }

    FunctionCall new_call = call;
    new_call.args = new_args;
    new_call.template_args = new_template_args;

    lowered.push_back(new const ComputeNode(new_call, node->getHeader(),
                                            current_adt.at(node->getAnnotation().getPattern().getOutput().getDS())));

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
    std::vector<LowerIR> lowered;
    for (const auto &rel : node->old_to_new) {
        all_relationships[rel.first] = rel.second;
    }
    auto tileable_fileds = node->getAnnotation().getPattern().getTileableFields();
    for (const auto &field : tileable_fileds) {
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }

    auto reducable_fields = node->getAnnotation().getPattern().getReducableFields();
    for (const auto &field : reducable_fields) {
        lowered.push_back(declare_intervals(get<0>(field.second), get<1>(field.second), field.first, get<2>(field.second)));
    }

    lowered.push_back(prepare_for_current_scope(node->staged_subset));
    this->visit(node->body);
    lowered.push_back(lowerIR);
    lowerIR = new const BlockNode(lowered);
}

}  // namespace gern
