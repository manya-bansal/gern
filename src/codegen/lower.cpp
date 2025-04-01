#include "codegen/lower.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "codegen/lower_visitor.h"
#include "utils/name_generator.h"

namespace gern {

void LowerIR::accept(LowerIRVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const LowerIR &n) {
    LowerPrinter print(os, 0);
    print.visit(n);
    return os;
}

void AllocateNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void FreeNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void InsertNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void GridDeclNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void SharedMemoryDeclNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void QueryNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void ComputeNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void IntervalNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void DefNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void AssertNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

bool IntervalNode::isMappedToGrid() const {
    return p != Grid::Unit::UNDEFINED;
}

Variable IntervalNode::getIntervalVariable() const {
    std::set<Variable> v = getVariables(start.getA());
    return *(v.begin());
}

void BlankNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

// void FunctionBoundary::accept(LowerIRVisitor *v) const {
//     v->visit(this);
// }

void OpaqueCall::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void BlockNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

LowerIR ComposableLower::lower() {
    // Add the inputs and output to the current scope.
    // these will come directly from the user.
    auto pattern = composable.getAnnotation().getPattern();
    AbstractDataTypePtr output = pattern.getOutput().getDS();
    current_ds.insert(output, output);

    auto inputs_subsets = pattern.getInputs();

    for (const auto &subset : inputs_subsets) {
        AbstractDataTypePtr input = subset.getDS();
        current_ds.insert(input, input);
    }

    this->visit(composable);
    return lowerIR;
}

LowerIR ComposableLower::generate_definitions(Assign definition) const {
    Variable v = to<Variable>(definition.getA());
    if (isConstExpr(v) && !isConstExpr(definition.getB())) {
        throw error::UserError("Binding const expr " + v.getName() + " to a non-const expr.");
    }
    if (v.isBound()) {
        throw error::UserError(v.getName() + " is determined by the pipeline and cannot be bound.");
    }
    return new const DefNode(definition, isConstExpr(definition.getB()));
}

LowerIR ComposableLower::generate_constraints(std::vector<Constraint> constraints) const {
    std::vector<LowerIR> lowered;
    for (const auto &c : constraints) {
        lowered.push_back(new const AssertNode(c));  // Need to add logic for lowering constraints.
    }
    return new const BlockNode(lowered);
}

template<typename T1, typename T2>
static T1 getFirstValue(const util::ScopedMap<T1, T1> &rel,
                        const util::ScopedMap<T2, T1> &map,
                        T1 entry) {
    while (rel.contains(entry)) {  // While we have the entry, go find it.
        if (map.contains(entry)) {
            return map.at(entry);  // Just set up the first value.
        }
        entry = rel.at(entry);
    }
    return T1();
}

void ComposableLower::lower(const TiledComputation *node) {
    // First, add the value of the captured value.
    Variable captured = node->captured;
    Variable loop_index = node->loop_index;
    auto old_to_new = node->old_to_new;
    std::vector<LowerIR> lowered;

    AbstractDataTypePtr output = node->getAnnotation().getPattern().getOutput().getDS();
    current_ds.scope();

    if (node->reduce) {
        current_ds.insert(output, getCurrent(output));
    }

    parents.scope();
    tiled_vars.scope();
    all_relationships.scope();
    tiled_dimensions.scope();
    staged_ds.scope();

    // Track all the relationships.
    for (const auto &var : old_to_new) {
        all_relationships.insert(var.first, var.second);
    }

    parents.insert(captured, node->v);
    tiled_dimensions.insert(node->parameter, node->v);

    tiled_vars.insert(captured, loop_index);

    std::cout << "Captured: " << captured << " Loop index: " << loop_index << std::endl;

    cur_tiled_vars.insert(loop_index);

    this->visit(node->tiled);  // Visit the actual object.

    for (const auto &v : cur_tiled_vars) {
        std::cout << "Cur tiled vars: " << v << std::endl;
    }

    cur_tiled_vars.extract(loop_index);

    current_ds.unscope();
    all_relationships.unscope();
    parents.unscope();
    tiled_vars.unscope();
    tiled_dimensions.unscope();
    staged_ds.unscope();

    Variable step_val = getFirstValue(all_relationships, parents, loop_index);
    bool has_parent = step_val.defined();
    lowerIR = new const IntervalNode(
        has_parent ? (loop_index = Expr(0)) : (loop_index = node->start),
        has_parent ? step_val : Expr(node->parameter),
        node->v,
        lowerIR,
        node->unit);
}

LowerIR ComposableLower::constructADTForCurrentScope(AbstractDataTypePtr d, std::vector<Expr> fields) {
    LowerIR ir = new const BlankNode();
    // If the data-stucture is in the current scope, skip.
    if (current_ds.contains_in_current_scope(d)) {
        return ir;
    }
    // If the adt is in the outer scope, generate a query.
    if (current_ds.contains(d)) {
        return constructQueryNode(d, fields);
    }

    // Otherwise generate an alloc.
    return constructAllocNode(d, fields);
}

template<typename T>
void ComposableLower::common(const T *node) {
    std::vector<LowerIR> lowered;
    Pattern pattern = node->getAnnotation().getPattern();
    SubsetObj output_subset = pattern.getOutput();
    AbstractDataTypePtr output = output_subset.getDS();
    std::vector<Expr> fields = output_subset.getFields();

    AbstractDataTypePtr parent;
    bool to_insert = false;

    if (!current_ds.contains_in_current_scope(output)) {
        assert(current_ds.contains(output));
        parent = getCurrent(output);       // Get the current parent before query over-writes it.
        to_insert = parent.insertQuery();  // Do we need to insert out query?
        lowered.push_back(constructQueryNode(output, output_subset.getFields()));
    }

    // Now that the output has been declared, visit the node.
    lower(node);
    lowered.push_back(lowerIR);
    // Free the actual node.
    // Step 4: Insert the final output.
    AbstractDataTypePtr child = getCurrent(output);
    if (to_insert) {
        lowered.push_back(constructInsertNode(parent, child, queried_with.at(parent)));
    }

    lowerIR = new const BlockNode(lowered);
}

void ComposableLower::lower(const Computation *node) {

    std::vector<LowerIR> lowered;

    std::vector<Composable> composed = node->composed;
    // Now, we are ready to lower the composable objects that make up the body.
    for (const auto &function : composed) {
        // Construct the allocations and track whether the allocations
        // need to be freed.
        SubsetObj temp_subset = function.getAnnotation().getPattern().getOutput();
        AbstractDataTypePtr temp = temp_subset.getDS();
        if (!current_ds.contains(temp)) {
            lowered.push_back(constructAllocNode(temp, temp_subset.getFields()));
        }
        this->visit(function);
        lowered.push_back(lowerIR);
    }

    lowerIR = new const BlockNode(lowered);
}

void ComposableLower::visit(const Computation *node) {
    std::vector<LowerIR> lowered;
    lowered.push_back(declare_computes(node->getAnnotation().getPattern()));
    lowered.push_back(declare_consumes(node->getAnnotation().getPattern()));
    for (const auto &decl : node->declarations) {
        lowered.push_back(generate_definitions(decl));
    }
    common(node);
    lowered.push_back(lowerIR);
    lowerIR = new const BlockNode(lowered);
}

void ComposableLower::visit(const TiledComputation *node) {

    if (node->reduce) {  // Declares the output.
        std::vector<LowerIR> lowered;
        Pattern pattern = node->getAnnotation().getPattern();
        SubsetObj output_subset = pattern.getOutput();
        AbstractDataTypePtr output = output_subset.getDS();
        if (!current_ds.contains_in_current_scope(output)) {
            lowered.push_back(declare_computes(pattern));  // Declare anything that's necessary for the queries.
        }
        common(node);
        lowered.push_back(lowerIR);
        lowerIR = new const BlockNode(lowered);
    } else {
        lower(node);
    }
}

void ComposableLower::visit(const ComputeFunctionCall *node) {

    if (node == nullptr) {
        throw error::InternalError("GOING TO SCREAM");
    }

    std::vector<LowerIR> lowered;

    // Generate the constraints now.
    lowered.push_back(generate_constraints(node->getAnnotation().getConstraints()));

    Pattern pattern = node->getAnnotation().getPattern();
    std::vector<SubsetObj> inputs = pattern.getInputs();
    // Query the inputs if they are not an intermediate of the current pipeline.
    for (auto const &input : inputs) {
        // If the current DS is not in scope
        // (at this point we are already declared
        // all the intermediates), generate a query.
        AbstractDataTypePtr input_ds = input.getDS();
        if (!current_ds.contains_in_current_scope(input_ds)) {
            if (current_ds.contains(input_ds)) {
                // Generate a query.
                lowered.push_back(constructQueryNode(input_ds,
                                                     input.getFields()));
                AbstractDataTypePtr queried = getCurrent(input_ds);
            } else {
                throw error::InternalError("How did we get here?");
            }
        }
    }

    // Now, generate the function call.
    FunctionCall call = node->getCall();
    std::vector<Argument> new_args;
    auto args = call.args;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg) &&
            current_ds.contains(to<DSArg>(arg)->getADTPtr())) {
            new_args.push_back(Argument(current_ds.at(to<DSArg>(arg)->getADTPtr())));
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
                                            current_ds.at(node->getAnnotation().getPattern().getOutput().getDS())));

    lowerIR = new const BlockNode(lowered);
}

void ComposableLower::visit(const GlobalNode *node) {
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

void ComposableLower::visit(const StageNode *node) {
    std::vector<LowerIR> lowered;
    // declare the computes for the body.
    lowered.push_back(declare_computes(node->getAnnotation().getPattern()));
    lowered.push_back(declare_consumes(node->getAnnotation().getPattern()));

    // Now, stage the input or output.
    if (!current_ds.contains_in_current_scope(node->adt)) {
        lowered.push_back(constructQueryNode(node->adt, node->staged_subset.getFields()));
    }
    // Track all the relationships.
    for (const auto &var : node->old_to_new) {
        all_relationships.insert(var.first, var.second);
    }

    // lower the body.
    this->visit(node->body);
    lowered.push_back(lowerIR);

    lowerIR = new const BlockNode(lowered);
}

AbstractDataTypePtr ComposableLower::getCurrent(AbstractDataTypePtr ds) const {
    if (current_ds.contains(ds)) {
        return current_ds.at(ds);
    }
    return ds;
}

const QueryNode *ComposableLower::constructQueryNode(AbstractDataTypePtr ds, std::vector<Expr> args) {
    AbstractDataTypePtr queried = DummyDS::make(getUniqueName("_query_" + ds.getName()), "auto", ds);
    auto fields = getCurrentFields(ds, args);
    FunctionCall f = constructFunctionCall(ds.getQueryFunction(), ds.getFields(), fields);
    auto cur_scope_ds = getCurrent(ds);
    f.output = Parameter(queried);
    current_ds.insert(ds, queried);
    std::vector<Expr> staged_vars{cur_tiled_vars.begin(), cur_tiled_vars.end()};
    staged_vars.insert(staged_vars.end(), args.begin(), args.end());
    staged_ds.insert(ds, staged_vars);
    queried_with.insert(ds, fields);
    MethodCall call = MethodCall(cur_scope_ds, f);
    return new const QueryNode(ds, queried, fields, call);
}

const InsertNode *ComposableLower::constructInsertNode(AbstractDataTypePtr parent,
                                                       AbstractDataTypePtr child,
                                                       std::vector<Expr> insert_args) const {
    FunctionCall f = constructFunctionCall(parent.getInsertFunction(),
                                           parent.getFields(),
                                           insert_args);
    f.args.push_back(child);
    MethodCall call = MethodCall(parent, f);
    return new const InsertNode(call);
}

const AllocateNode *ComposableLower::constructAllocNode(AbstractDataTypePtr ds, std::vector<Expr> alloc_args) {
    auto fields = getCurrentFields(ds, alloc_args);
    FunctionCall alloc_func = constructFunctionCall(ds.getAllocateFunction(),
                                                    ds.getFields(),
                                                    fields);
    alloc_func.output = Parameter(ds);
    current_ds.insert(ds, ds);
    std::vector<Expr> staged_vars{cur_tiled_vars.begin(), cur_tiled_vars.end()};
    staged_vars.insert(staged_vars.end(), alloc_args.begin(), alloc_args.end());
    staged_ds.insert(ds, staged_vars);
    return new const AllocateNode(alloc_func);
}

template<typename T1>
static Expr getValue(const util::ScopedMap<T1, T1> &rel,
                     const util::ScopedMap<T1, T1> &map,
                     T1 entry,
                     std::set<Variable> stop_at = {}) {

    Expr e;
    bool encountered_stop_at = false;
    while (rel.contains(entry)) {  // While we have the entry, go find it.
        if (stop_at.contains(entry)) {
            encountered_stop_at = true;
        }
        if (map.contains(entry)) {
            if (encountered_stop_at) {
                return 0;
            }
            e = map.at(entry);  // Just set up the first value.
            entry = rel.at(entry);
            break;
        } else {
            entry = rel.at(entry);
        }
    }

    while (rel.contains(entry) && !stop_at.contains(entry)) {  // While we have the entry, go find it.
        if (map.contains(entry)) {
            e = e + map.at(entry);
        }
        entry = rel.at(entry);
    }
    return e;
}

std::vector<Expr> ComposableLower::getCurrentFields(AbstractDataTypePtr ds,
                                                    std::vector<Expr> true_md_fields) const {

    std::vector<Expr> new_true_md_fields = true_md_fields;

    if (staged_ds.contains(ds)) {  // if the data-structure has been staged.
        std::map<Variable, Expr> offset_by;
        std::set<Variable> staged_at;

        auto staged_with = staged_ds.at(ds);  // Get the variables at which level the data-structure was staged.

        for (const auto &staged : staged_with) {
            auto vars = getVariables(staged);
            for (const auto &v : vars) {
                staged_at.insert(v);
            }
        }

        for (size_t i = 0; i < true_md_fields.size(); i++) {
            auto vars = getVariables(true_md_fields[i]);
            for (const auto &v : vars) {
                if (!offset_by.contains(v)) {
                    Expr e = getValue(all_relationships, tiled_vars, v, staged_at);  // Get the value of the variable at the level of the staged data-structure.
                    if (e.defined()) {
                        offset_by[v] = e;
                    }
                }
            }
            new_true_md_fields[i] = replaceVariables(true_md_fields[i], offset_by);  // Replace the variables with their offset values.
        }
    }
    return new_true_md_fields;
}

FunctionCall ComposableLower::constructFunctionCall(FunctionSignature f,
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

LowerIR ComposableLower::declare_computes(Pattern annotation) const {
    std::vector<LowerIR> lowered;
    match(annotation, std::function<void(const ComputesForNode *, Matcher *)>(
                          [&](const ComputesForNode *op, Matcher *ctx) {
                              ctx->match(op->body);
                              Variable v = to<Variable>(op->start.getA());
                              Expr rhs = getValue(all_relationships, tiled_vars, v);
                              if (rhs.defined()) {
                                  lowered.push_back(generate_definitions(v = rhs));
                              } else {
                                  lowered.push_back(
                                      new const DefNode(op->start, false));
                              }
                              Variable step_val = getFirstValue(all_relationships, parents, v);
                              if (step_val.defined()) {
                                  lowered.push_back(
                                      generate_definitions(op->step = step_val));
                              } else {
                                  lowered.push_back(
                                      generate_definitions(op->step = op->parameter));
                              }
                          }));
    return new const BlockNode(lowered);
}

LowerIR ComposableLower::declare_consumes(Pattern annotation) const {
    std::vector<LowerIR> lowered;
    match(annotation, std::function<void(const ConsumesForNode *, Matcher *)>(
                          [&](const ConsumesForNode *op, Matcher *ctx) {
                              ctx->match(op->body);
                              Variable v = to<Variable>(op->start.getA());
                              Expr rhs = getValue(all_relationships, tiled_vars, v);
                              if (rhs.defined()) {
                                  lowered.push_back(generate_definitions(v = rhs));
                              } else {
                                  lowered.push_back(
                                      new const DefNode(op->start, false));
                              }
                              Variable step_val = getFirstValue(all_relationships, parents, v);
                              if (step_val.defined()) {
                                  lowered.push_back(
                                      generate_definitions(op->step = step_val));
                              } else {
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->step = op->parameter));
                              }
                          }));
    return new const BlockNode(lowered);
}

}  // namespace gern