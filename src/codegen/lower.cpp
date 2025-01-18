#include "codegen/lower.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/lower_visitor.h"
#include "compose/pipeline.h"

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

LowerIR ComposeLower::lower() {
    if (has_been_lowered) {
        return final_lower;
    }
    visit(c);
    has_been_lowered = true;
    return final_lower;
}

void ComposeLower::visit(const ComputeFunctionCall *c) {
    // Generate the output query if it not an intermediate.
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> queries;
    AbstractDataTypePtr output = c->getOutput();
    Argument queried;

    std::vector<LowerIR> lowered_func;
    if (!isIntermediate(output)) {
        const QueryNode *q = constructQueryNode(output,
                                                c->getMetaDataFields(output));
        queried = q->f.output;
        lowered_func.push_back(q);
    }

    // Generate the queries for the true inputs.
    // since intermediates should have been allocated,
    // Putting these in temp because it may need to wrapped
    // in an interval node.
    std::set<AbstractDataTypePtr> func_inputs = c->getInputs();
    std::vector<LowerIR> temp;
    for (const auto &in : func_inputs) {
        if (!isIntermediate(in)) {
            // Generate the query node.
            temp.push_back(constructQueryNode(in,
                                              c->getMetaDataFields(in)));
        }
    }

    // Rewrite the call with the queries and actually construct the compute node.
    FunctionCall new_call = c->getCall().replaceAllDS(new_ds);
    temp.push_back(new const ComputeNode(new_call, c->getHeader()));

    // Now generate all the consumer intervals if any!
    LowerIR intervals = generateConsumesIntervals(c, temp);
    lowered_func.push_back(intervals);

    // Insert the computed output if necessary.
    if (!isIntermediate(output) && output.insertQuery()) {
        FunctionCall f = constructFunctionCall(output.getInsertFunction(),
                                               output.getFields(), c->getMetaDataFields(output));
        f.name = output.getName() + "." + f.name;
        f.args.push_back(Argument(queried));
        lowered_func.push_back(new InsertNode(output, f));
    }

    final_lower = new const BlockNode(lowered_func);
}

bool ComposeLower::isIntermediate(AbstractDataTypePtr d) const {
    return intermediates_set.contains(d);
}

void ComposeLower::visit(const PipelineNode *node) {
    intermediates_set = node->p.getIntermediates();
    std::vector<LowerIR> lowered;
    // Generate all the variable definitions
    // that the functions can then directly refer to.
    std::vector<LowerIR> ir_nodes = generateAllDefs(node);
    lowered.insert(lowered.end(), ir_nodes.begin(), ir_nodes.end());
    // Generate all the allocations.
    // For nested pipelines, this should
    // generated nested allocations for
    // temps as well.
    ir_nodes = generateAllAllocs(node);
    lowered.insert(lowered.end(), ir_nodes.begin(), ir_nodes.end());
    // Rewrite functions to use the allocated data-structures.
    std::vector<Compose> compose_w_allocs = rewriteCalls(node);
    // Generates all the queries and the compute node.
    // by visiting all the functions.
    for (const auto &compose : compose_w_allocs) {
        // There HAS to be a better way, this is so
        // ugly.
        if (isa<PipelineNode>(compose)) {
            ComposeLower child_lower(compose);
            LowerIR child_ir = child_lower.lower();
            // Generate the queries
            // ir_nodes = generateAllChildQueries(node, to<PipelineNode>(compose));
            // lowered.insert(lowered.end(), ir_nodes.begin(), ir_nodes.end());
            lowered.push_back(new const FunctionBoundary(child_ir));
        } else {
            this->visit(compose);
            lowered.push_back(final_lower);
        }
    }
    // Generate all the free nodes now.
    ir_nodes = generateAllFrees(node);
    lowered.insert(lowered.end(), ir_nodes.begin(), ir_nodes.end());
    final_lower = generateOuterIntervals(node->p.getProducerFunction(node->p.getOutput()), lowered);
}

std::vector<LowerIR> ComposeLower::generateAllDefs(const PipelineNode *node) {
    std::vector<LowerIR> lowered;
    std::vector<Assign> definitions = node->p.getDefinitions();
    for (const auto &def : definitions) {
        Variable v = to<Variable>(def.getA());
        if (v.isBound()) {
            throw error::UserError("Trying to bind a variable that has already been bound");
        }
        lowered.push_back(new const DefNode(def,
                                            node->isTemplateArg(v)));
    }
    return lowered;
}

std::vector<LowerIR> ComposeLower::generateAllAllocs(const PipelineNode *node) {
    // We need to define an allocation for all the
    // intermediates.
    std::vector<LowerIR> lowered;
    std::vector<SubsetObj> all_input_subsets = node->getAnnotation().getAllConsumesSubsets();
    for (const auto &subset : all_input_subsets) {
        if (node->p.isIntermediate(subset.getDS())) {
            AbstractDataTypePtr adt = subset.getDS();
            lowered.push_back(constructAllocNode(
                adt,
                subset.getFields()));
        }
    }
    return lowered;
}

std::vector<Compose> ComposeLower::rewriteCalls(const PipelineNode *node) const {
    std::vector<Compose> new_funcs;
    for (const auto &c : node->p.getFuncs()) {
        new_funcs.push_back(c.replaceAllDS(new_ds));
    }
    return new_funcs;
}

// std::vector<LowerIR> ComposeLower::generateAllQueries(const PipelineNode *node) {
//     std::set<AbstractDataTypePtr> child_inputs = node->p.getInputs();
// }

std::vector<LowerIR> ComposeLower::generateAllFrees(const PipelineNode *) {
    std::vector<LowerIR> lowered;
    for (const auto &ds : to_free) {
        lowered.push_back(constructFreeNode(ds));
    }
    return lowered;
}

const FreeNode *ComposeLower::constructFreeNode(AbstractDataTypePtr ds) {
    return new FreeNode(ds);
}

const AllocateNode *ComposeLower::constructAllocNode(AbstractDataTypePtr ds, std::vector<Expr> alloc_args) {
    FunctionCall alloc_func = constructFunctionCall(ds.getAllocateFunction(), ds.getFields(), alloc_args);
    alloc_func.output = Parameter(ds);
    if (ds.freeAlloc()) {
        to_free.insert(ds);
    }
    return new AllocateNode(alloc_func);
}

const QueryNode *ComposeLower::constructQueryNode(AbstractDataTypePtr ds, std::vector<Expr> query_args) {
    AbstractDataTypePtr queried = PipelineDS::make(getUniqueName("_query_" + ds.getName()), "auto", ds);
    FunctionCall f = constructFunctionCall(ds.getQueryFunction(), ds.getFields(), query_args);
    f.name = ds.getName() + "." + f.name;
    f.output = Parameter(queried);
    new_ds[ds] = queried;
    // If any of the queried data-structures need to be free, track that.
    if (ds.freeQuery()) {
        to_free.insert(queried);
    }
    return new QueryNode(ds, f);
}

FunctionCall ComposeLower::constructFunctionCall(FunctionSignature f, std::vector<Variable> ref_md_fields, std::vector<Expr> true_md_fields) const {

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

LowerIR ComposeLower::generateConsumesIntervals(ComputeFunctionCallPtr f, std::vector<LowerIR> body) const {
    LowerIR current = new const BlockNode(body);
    match(f->getAnnotation(), std::function<void(const ConsumesForNode *, Matcher *)>(
                                  [&](const ConsumesForNode *op, Matcher *ctx) {
                                      ctx->match(op->body);
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

LowerIR ComposeLower::generateOuterIntervals(ComputeFunctionCallPtr f, std::vector<LowerIR> body) const {
    LowerIR current = new const BlockNode(body);
    match(f->getAnnotation(), std::function<void(const ComputesForNode *, Matcher *)>(
                                  [&](const ComputesForNode *op, Matcher *ctx) {
                                      ctx->match(op->body);
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
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
    return getIntervalVariable().isBoundToGrid();
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

}  // namespace gern