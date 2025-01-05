#ifndef GERN_PIPELINE_NODES_H
#define GERN_PIPELINE_NODES_H

#include "annotations/data_dependency_language.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

typedef const FunctionCall *FunctionCallPtr;
class PipelineVisitor;
struct PipelineNode;

using GernGenFuncPtr = void (*)(void **);

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
    Pipeline(std::vector<Compose> compose)
        : compose(compose) {
    }

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
    void compile(std::string compile_flags = "");
    // This function actually runs the function pointer, and compiles
    // the function pointer, if it hasn't already been compiled. Currently
    // it takes adts and variables separately, I may need to extend this handle
    // other types as well.
    void evaluate(std::map<std::string, void *> args);

    using CompositionVisitor::visit;

    void visit(const FunctionCall *c);
    void visit(const PipelineNode *c);

    std::vector<LowerIR> getIRNodes() const;
    std::map<Variable, Expr> getVariableDefinitions() const;

    void accept(CompositionVisitor *) const;

private:
    bool isIntermediate(AbstractDataTypePtr d) const;
    std::vector<Expr> generateMetaDataFields(AbstractDataTypePtr, FunctionCallPtr) const;
    std::vector<LowerIR> generateConsumesIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<LowerIR> generateOuterIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<Compose> compose;
    std::vector<LowerIR> lowered;
    std::map<Variable, Expr> variable_definitions;
    bool compiled = false;
    GernGenFuncPtr fp;
    std::vector<std::string> argument_order;
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
    IntervalNode(Assign start, Constraint end, Assign step, std::vector<LowerIR> body)
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
    Constraint end;
    Assign step;
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

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct BlankNode : public LowerIRNode {
    BlankNode() = default;
    void accept(PipelineVisitor *) const;
};

}  // namespace gern

#endif