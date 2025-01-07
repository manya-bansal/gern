#pragma once

#include "annotations/data_dependency_language.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

class PipelineVisitor;
struct PipelineNode;

using FunctionCallPtr = const FunctionCall *;
using Dataflow = std::map<AbstractDataTypePtr, std::set<AbstractDataTypePtr>>;
using OutputFunction = std::map<AbstractDataTypePtr, FunctionCallPtr>;

struct LowerIRNode : public util::Manageable<LowerIRNode>,
                     public util::Uncopyable {
    LowerIRNode() = default;
    virtual ~LowerIRNode() = default;
    virtual void accept(PipelineVisitor *) const = 0;
};

class LowerIR : public util::IntrusivePtr<const LowerIRNode> {
public:
    LowerIR()
        : util::IntrusivePtr<const LowerIRNode>(nullptr) {
    }
    LowerIR(const LowerIRNode *n)
        : util::IntrusivePtr<const LowerIRNode>(n) {
    }
    void accept(PipelineVisitor *v) const;
};

std::ostream &operator<<(std::ostream &os, const LowerIR &n);

// The pipeline actually holds the lowered
// nodes, and helps us nest pipelines.
class Pipeline : public CompositionVisitor {

public:
    Pipeline(std::vector<Compose> compose);

    std::vector<Compose> getFuncs() const {
        return compose;
    }

    /**
     * @brief Returns the total number of functions
     *        including those in nested pipelines.
     *
     * @return int
     */
    int numFuncs();
    Pipeline &at_device();
    Pipeline &at_host();
    bool is_at_device() const;

    void lower();

    using CompositionVisitor::visit;
    void visit(const FunctionCall *c);
    void visit(const PipelineNode *c);

    std::vector<LowerIR> getIRNodes() const;
    std::map<Variable, Expr> getVariableDefinitions() const;
    // Returns the function call that produces a particular output.
    FunctionCallPtr getProducerFunc(AbstractDataTypePtr) const;
    std::set<AbstractDataTypePtr> getInputs() const;
    AbstractDataTypePtr getOutput() const;
    OutputFunction getOutputFunctions() const;

    void accept(CompositionVisitor *) const;

    Dataflow getDataflow() const;

private:
    void init(std::vector<Compose> compose);
    bool isIntermediate(AbstractDataTypePtr d) const;

    std::vector<Expr> generateMetaDataFields(AbstractDataTypePtr, FunctionCallPtr);
    std::vector<LowerIR> generateConsumesIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<LowerIR> generateOuterIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<Compose> compose;
    std::vector<LowerIR> lowered;
    std::map<Variable, Expr> variable_definitions;
    Dataflow dataflow;                                          // Tracks the output to inputs relationships.
    OutputFunction output_function;                             // Tracks what function produces an output data-structure.
    AbstractDataTypePtr final_output;                           // Tracks the final output that gets produced.
    std::set<AbstractDataTypePtr> all_inputs;                   // Tracks the data-structures that have been used as inputs.
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;  // New names for the ds.
    std::set<AbstractDataTypePtr> to_free;                      // New names for the ds.
    bool has_been_lowered = false;
    bool device = false;
};

std::ostream &operator<<(std::ostream &os, const Pipeline &p);

// IR Node that marks an allocation
struct AllocateNode : public LowerIRNode {
    AllocateNode(AbstractDataTypePtr data, const std::vector<Expr> &fields)
        : data(data), fields(fields) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr data;
    std::vector<Expr> fields;
};

// IR Node that marks an free
struct FreeNode : public LowerIRNode {
    FreeNode(AbstractDataTypePtr data)
        : data(data) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr data;
};

// IR Node that marks an insertion
// The child data structure is inserted
// into the parent data-structure as the
// subset with meta-data values in fields.
struct InsertNode : public LowerIRNode {
    InsertNode(AbstractDataTypePtr parent, AbstractDataTypePtr child)
        : parent(parent), child(child) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr parent;
    AbstractDataTypePtr child;
    std::vector<Expr> fields;
};

// IR Node that marks a query
// The child data structure is produced
// from the parent data-structure corresponding to
// the subset with meta-data values in fields.
struct QueryNode : public LowerIRNode {
    QueryNode(AbstractDataTypePtr parent, AbstractDataTypePtr child, std::vector<Expr> fields)
        : parent(parent), child(child), fields(fields) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr parent;
    AbstractDataTypePtr child;
    std::vector<Expr> fields;
};

// IR Node marks a function call.
struct ComputeNode : public LowerIRNode {
    ComputeNode(FunctionCallPtr f, std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds)
        : f(f), new_ds(new_ds) {
    }
    void accept(PipelineVisitor *) const;
    FunctionCallPtr f;
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
};

// To track the interval nodes that need to be generated
// for the pipeline.
struct IntervalNode : public LowerIRNode {
public:
    IntervalNode(Assign start, Expr end, Expr step, std::vector<LowerIR> body)
        : start(start), end(end), step(step), body(body) {
    }
    void accept(PipelineVisitor *) const;

    /**
     * @brief isMappedToGrid returns whether the interval
     *        is mapped to the grid of the GPU, i.e.
     *        whether the interval variable is bound
     *        to a GPU property.
     *
     */
    bool isMappedToGrid() const;
    Variable getIntervalVariable() const;

    Assign start;
    Expr end;
    Expr step;
    std::vector<LowerIR> body;
};

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct PipelineNode : public CompositionObject {
    PipelineNode(Pipeline p)
        : p(p) {
    }

    void accept(CompositionVisitor *) const;
    Pipeline p;
};

// Node to declare definitions of variables.
struct DefNode : public LowerIRNode {
    DefNode(Assign assign)
        : assign(assign) {
    }

    void accept(PipelineVisitor *) const;
    Assign assign;
};

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct BlankNode : public LowerIRNode {
    BlankNode() = default;
    void accept(PipelineVisitor *) const;
};

}  // namespace gern