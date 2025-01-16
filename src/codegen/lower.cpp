#include "codegen/lower.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "compose/pipeline.h"
#include "compose/pipeline_visitor.h"

namespace gern {

void LowerIR::accept(LowerIRVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const LowerIR &n) {
    PipelinePrinter print(os, 0);
    print.visit(n);
    return os;
}

std::vector<LowerIR> ComposeLower::lower() {
    if (has_been_lowered) {
        return lowered;
    }
    visit(c);
    has_been_lowered = true;
    return lowered;
}

void ComposeLower::visit(const ComputeFunctionCall *c) {
    // Generate the output query if it not an intermediate.
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> queries;
    AbstractDataTypePtr output = c->getOutput();
    Argument queried;
    if (!isIntermediate(output)) {
        const QueryNode *q = constructQueryNode(output,
                                                c->getMetaDataFields(output));
        queried = q->f.output;
        lowered.push_back(q);
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
    std::vector<LowerIR> intervals = generateConsumesIntervals(c, temp);
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());

    // Insert the computed output if necessary.
    if (!isIntermediate(output) && output.insertQuery()) {
        FunctionCall f = constructFunctionCall(output.getInsertFunction(),
                                               output.getFields(), c->getMetaDataFields(output));
        f.name = output.getName() + "." + f.name;
        f.args.push_back(Argument(queried));
        lowered.push_back(new InsertNode(output, f));
    }
}

void ComposeLower::visit(const PipelineNode *node) {
    // Generate all the variable definitions
    // that the functions can then directly refer to.
    generateAllDefs(node);
    // Generate all the allocations.
    // For nested pipelines, this should
    // generated nested allocations for
    // temps as well.
    std::vector<Compose> compose_w_allocs = generateAllAllocs(node);
    // Generates all the queries and the compute node.
    // by visiting all the functions.
    for (const auto &funcs : compose_w_allocs) {
        this->visit(funcs);
    }
    // Generate all the free nodes now.
    generateAllFrees(node);

    // If its a vec of vec...then, need to do something else.
    // right now, gern will complain if you try.
    // Now generate the outer loops.
    lowered = generateOuterIntervals(
        to<ComputeFunctionCall>(compose[compose.size() - 1].ptr), lowered);
}

bool ComposeLower::isIntermediate(AbstractDataTypePtr d) const {
    return intermediates_set.contains(d);
}

void ComposeLower::generateAllDefs(const PipelineNode *node) {
    std::vector<AbstractDataTypePtr> intermediates = node->p.getAllOutputs();
    if (intermediates.size() <= 1) {
        // Do nothing.
        return;
    }
    auto it = intermediates.rbegin();
    it++;  // Skip the last one.
    // Go in reverse order, and define all the variables.
    for (; it < intermediates.rend(); ++it) {
        // Define the variables assosciated with the producer func.
        // For all functions that consume this data-structure, figure out the relationships for
        // this input.
        AbstractDataTypePtr temp = *it;
        ComputeFunctionCallPtr producer_func = node->p.getProducerFunction(temp);
        std::set<ComputeFunctionCallPtr> consumer_funcs = node->p.getConsumerFunctions(temp);
        // No forks allowed, as is. Come back and change!
        if (consumer_funcs.size() != 1) {
            std::cout << "WARNING::FORKS ARE NOT IMPL, ASSUMING EQUAL CONSUMPTION!" << std::endl;
        }
        // Writing as a for loop, will eventually change!
        // Only one allowed for right now...
        std::vector<Variable> var_fields = producer_func->getProducesFields();
        for (const auto &cf : consumer_funcs) {
            std::vector<Expr> consumer_fields = cf->getMetaDataFields(temp);
            if (consumer_fields.size() != var_fields.size()) {
                throw error::InternalError("Annotations for " + cf->getName() + " and " + producer_func->getName() +
                                           " do not have the same size for " + temp.getName());
            }
            for (size_t i = 0; i < consumer_fields.size(); i++) {
                if (var_fields[i].isBound()) {
                    throw error::UserError(var_fields[i].getName() +
                                           " while producing " + temp.getName() +
                                           " is completely bound, but is being set.");
                }
                lowered.push_back(new DefNode(var_fields[i] = consumer_fields[i],
                                              producer_func->isTemplateArg(var_fields[i])));
            }
            break;
        }
    }
}

std::vector<Compose> ComposeLower::generateAllAllocs(const PipelineNode *node) {
    // We need to define an allocation for all the
    // intermediates.
    std::vector<AbstractDataTypePtr> intermediates = node->p.getAllOutputs();
    for (size_t i = 0; i < intermediates.size() - 1; i++) {
        AbstractDataTypePtr temp = intermediates[i];
        // Finally make the allocation.
        lowered.push_back(constructAllocNode(
            temp,
            node->p.getProducerFunction(temp)->getMetaDataFields(temp)));
    }

    // Change references to the allocated data-structures now.
    std::vector<Compose> new_funcs;
    for (const auto &c : node->p.getFuncs()) {
        new_funcs.push_back(c.replaceAllDS(new_ds));
    }
    return new_funcs;
}

void ComposeLower::generateAllFrees(const PipelineNode *) {
    for (const auto &ds : to_free) {
        lowered.push_back(constructFreeNode(ds));
    }
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
    AbstractDataTypePtr queried = PipelineDS::make(getUniqueName("_query_" + ds.getName()), ds);
    FunctionCall f = constructFunctionCall(ds.getQueryFunction(), ds.getFields(), query_args);
    f.name = ds.getName() + "." + f.name;
    f.output = Argument(queried);
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

std::vector<LowerIR> ComposeLower::generateConsumesIntervals(ComputeFunctionCallPtr f, std::vector<LowerIR> body) const {
    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ConsumesForNode *, Matcher *)>(
                                  [&](const ConsumesForNode *op, Matcher *ctx) {
                                      ctx->match(op->body);
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

std::vector<LowerIR> ComposeLower::generateOuterIntervals(ComputeFunctionCallPtr f, std::vector<LowerIR> body) const {
    std::vector<LowerIR> current = body;
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

}  // namespace gern