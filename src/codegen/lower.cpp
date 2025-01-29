#include "codegen/lower.h"
#include "annotations/lang_nodes.h"
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

bool IntervalNode::isMappedToGrid() const {
    return p != Grid::Property::UNDEFINED;
}

Variable IntervalNode::getIntervalVariable() const {
    std::set<Variable> v = getVariables(start.getA());
    return *(v.begin());
}

void BlankNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void FunctionBoundary::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void BlockNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

LowerIR ComposableLower::lower() {
    this->visit(composable);
    return lowerIR;
}

void ComposableLower::common(const ComposableNode *node) {
    if (node == nullptr) {
        throw error::InternalError("GOING TO SCREAM");
    }

    std::vector<LowerIR> lowered;
    // Declare any reductions if present.
    lowered.push_back(declare_consumes(node->getAnnotation()));
    std::set<AbstractDataTypePtr> to_free;  // Track all the data-structures that need to be freed.
    Pattern annotation = node->getAnnotation();
    std::vector<SubsetObj> inputs = annotation.getInputs();
    // Declare all the outer variables.
    // Query the inputs if they are not an intermediate of the current pipeline.
    for (auto const &input : inputs) {
        // If the current DS is not in scope
        // (at this point we are already declared
        // all the intermediates), generate a query.
        AbstractDataTypePtr input_ds = input.getDS();
        if (!intermediates.contains_at_current_scope(input_ds)) {
            // Generate a query.
            lowered.push_back(constructQueryNode(input_ds,
                                                 input.getFields()));
            AbstractDataTypePtr queried = getCurrent(input_ds);
            if (queried.freeQuery()) {
                to_free.insert(queried);
            }
        }
    }

    // Now, visit the node.
    this->visit(node);
    lowered.push_back(lowerIR);

    // Free any the queried subsets.
    for (const auto &ds : to_free) {
        lowered.push_back(new const FreeNode(ds));
    }

    lowerIR = new const BlockNode(lowered);
}

LowerIR ComposableLower::generate_definitions(Assign definition) const {
    Variable v = to<Variable>(definition.getA());
    if (isConstExpr(v) && !isConstExpr(definition.getB())) {
        throw error::UserError("Binding const expr " + v.getName() + " to a non-const expr.");
    }
    if (v.isBound()) {
        throw error::UserError(v.getName() + " is determined by the pipeline and cannot be bound.");
    }
    return new const DefNode(definition, v.isConstExpr());
}

void ComposableLower::visit(const Computation *node) {

    std::vector<LowerIR> lowered;           // Vector for lowered code.
    std::set<AbstractDataTypePtr> to_free;  // Track all the data-structures that need to be freed.

    // Step 0: Declare all the variable relationships.
    lowered.push_back(declare_computes(node->getAnnotation()));
    for (const auto &decl : node->declarations) {
        lowered.push_back(generate_definitions(decl));
    }

    // Step 1 : Query the output and track whether the query needs to be
    // freed.
    SubsetObj output_subset = node->getAnnotation().getOutput();
    AbstractDataTypePtr output = output_subset.getDS();
    std::vector<Expr> fields = output_subset.getFields();
    AbstractDataTypePtr parent = getCurrent(output);  // Get the current parent before query over-writes it.
    lowered.push_back(constructQueryNode(output, output_subset.getFields()));
    AbstractDataTypePtr child = getCurrent(output);
    if (child.freeQuery()) {
        to_free.insert(child);
    }

    // Step 2: Construct the allocations and track whether the allocations
    // need to be freed.
    std::vector<Composable> composed = node->composed;
    size_t size_funcs = composed.size();
    for (size_t i = 0; i < size_funcs - 1; i++) {
        SubsetObj temp_subset = composed[i].getAnnotation().getOutput();
        AbstractDataTypePtr temp = temp_subset.getDS();
        lowered.push_back(constructAllocNode(temp, temp_subset.getFields()));
        intermediates.insert(temp);
        // Do we need to free our allocs?
        if (temp.freeAlloc()) {
            to_free.insert(temp);
        }
    }

    // std::vector<LowerIR> body;
    // Step 3: Now, we are ready to lower the composable objects that make up the body.
    for (const auto &function : node->composed) {
        common(function.ptr);
        lowered.push_back(lowerIR);
    }

    // Step 4: Insert the final output.
    if (parent.insertQuery()) {
        lowered.push_back(constructInsertNode(parent, child, fields));
    }

    for (const auto &ds : to_free) {
        lowered.push_back(new const FreeNode(ds));
    }

    lowerIR = new const BlockNode(lowered);
}

LowerIR ComposableLower::declare_computes(Pattern annotation) const {
    std::vector<LowerIR> lowered;
    match(annotation, std::function<void(const ComputesForNode *, Matcher *)>(
                          [&](const ComputesForNode *op, Matcher *ctx) {
                              ctx->match(op->body);
                              Variable v = to<Variable>(op->start.getA());
                              if (tiled_vars.contains(v)) {
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(v = tiled_vars.at(v)));
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->step = parents.at(op->step)));
                              } else {
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->start));
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->step = op->end));
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
                              if (tiled_vars.contains(v)) {
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(v = tiled_vars.at(v)));
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->step = parents.at(op->step)));
                              } else {
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->start));
                                  lowered.insert(lowered.begin(),
                                                 generate_definitions(op->step = op->end));
                              }
                          }));
    return new const BlockNode(lowered);
}

void ComposableLower::visit(const TiledComputation *node) {

    std::vector<LowerIR> lowered;
    // First, add the value of the captured value.
    Variable captured = node->captured;
    Variable loop_index(getUniqueName("_i_"));

    // Track the fact that this field is being tiled.
    Expr current_value = loop_index;
    if (tiled_vars.contains(captured)) {
        current_value = current_value + tiled_vars.at(captured);
    }

    // Next, lower the composable object.
    current_ds.scope();
    parents.scope();
    tiled_vars.scope();
    parents.insert(node->step, node->v);
    tiled_vars.insert(captured, current_value);
    intermediates.scope();
    this->visit(node->tiled);
    current_ds.unscope();
    parents.unscope();
    tiled_vars.unscope();
    intermediates.unscope();

    bool has_parent = parents.contains(node->step);
    lowerIR = new const IntervalNode(
        has_parent ? (loop_index = Expr(0)) : (loop_index = node->start),
        has_parent ? Expr(parents.at(node->step)) : Expr(node->end),
        node->v,
        lowerIR,
        node->property);
}

AbstractDataTypePtr ComposableLower::getCurrent(AbstractDataTypePtr ds) const {
    if (current_ds.contains(ds)) {
        return current_ds.at(ds);
    }
    return ds;
}
// Constructs a query node for a data-structure, and tracks this relationship.
const QueryNode *ComposableLower::constructQueryNode(AbstractDataTypePtr ds, std::vector<Expr> args) {

    AbstractDataTypePtr ds_in_scope = getCurrent(ds);
    AbstractDataTypePtr queried = DummyDS::make(getUniqueName("_query_" + ds_in_scope.getName()), "auto", ds);
    FunctionCall f = constructFunctionCall(ds.getQueryFunction(), ds_in_scope.getFields(), args);
    f.name = ds_in_scope.getName() + "." + f.name;
    f.output = Parameter(queried);
    current_ds.insert(ds, queried);
    return new const QueryNode(ds, f);
}

const InsertNode *ComposableLower::constructInsertNode(AbstractDataTypePtr parent,
                                                       AbstractDataTypePtr child,
                                                       std::vector<Expr> insert_args) const {
    FunctionCall f = constructFunctionCall(parent.getInsertFunction(),
                                           parent.getFields(),
                                           insert_args);
    f.name = parent.getName() + "." + f.name;
    f.args.push_back(child);
    return new const InsertNode(parent, f);
}

const AllocateNode *ComposableLower::constructAllocNode(AbstractDataTypePtr ds, std::vector<Expr> alloc_args) {
    FunctionCall alloc_func = constructFunctionCall(ds.getAllocateFunction(), ds.getFields(), alloc_args);
    alloc_func.output = Parameter(ds);

    return new const AllocateNode(alloc_func);
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
        new_args.push_back(Argument(mappings[to<VarArg>(a)->getVar()]));
    }
    // set up the templated args.
    std::vector<Expr> template_args;
    for (auto const &a : f.template_args) {
        template_args.push_back(mappings[a]);
    }

    FunctionCall f_new{
        .name = f.name,
        .args = new_args,
        .template_args = template_args,
    };

    return f_new;
}

void ComposableLower::visit(const ComputeFunctionCall *node) {
    FunctionCall call = node->getCall();
    std::vector<Argument> new_args;

    auto args = call.args;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg) &&
            current_ds.contains(to<DSArg>(arg)->getADTPtr())) {
            new_args.push_back(Argument(current_ds.at(to<DSArg>(arg)->getADTPtr())));
        } else {
            new_args.push_back(arg);
        }
    }

    FunctionCall new_call{
        .name = call.name,
        .args = new_args,
        .template_args = call.template_args,
        .output = call.output,
    };

    lowerIR = new const ComputeNode(new_call, node->getHeader());
}

}  // namespace gern