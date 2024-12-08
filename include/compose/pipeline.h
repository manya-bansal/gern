#ifndef GERN_PIPELINE_NODES_H
#define GERN_PIPELINE_NODES_H

#include "annotations/data_dependency_language.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

typedef const FunctionCall *FunctionCallPtr;

struct LowerIRNode : public util::Manageable<LowerIRNode>,
                     public util::Uncopyable {
    LowerIRNode() = default;
    virtual ~LowerIRNode() = default;
};

class LowerIR : public util::IntrusivePtr<const LowerIRNode> {
public:
    LowerIR()
        : util::IntrusivePtr<const LowerIRNode>(nullptr) {
    }
    LowerIR(const LowerIRNode *n)
        : util::IntrusivePtr<const LowerIRNode>(n) {
    }
};

// The pipeline actually holds the lowered
// nodes, and helps us nest pipelines.
class Pipeline : public LowerIRNode,
                 public CompositionVisitor {
public:
    Pipeline(std::vector<Compose> compose)
        : compose(compose) {
    }

    void lower();
    using CompositionVisitor::visit;
    void visit(const FunctionCall *c);
    void visit(const ComposeVec *c);
    std::vector<LowerIR> getIRNodes();

private:
    bool isIntermediate(AbstractDataTypePtr d) const;
    std::vector<Expr> generateMetaDataFields(AbstractDataTypePtr, FunctionCallPtr) const;
    std::vector<LowerIR> generateConsumesIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<LowerIR> generateOuterIntervals(FunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<Compose> compose;
    std::vector<LowerIR> lowered;
};

// IR Node that marks an allocation
struct AllocateNode : public LowerIRNode {
    AllocateNode(AbstractDataTypePtr data, const std::vector<Expr> &fields)
        : data(data), fields(fields) {
    }
    AbstractDataTypePtr data;
    std::vector<Expr> fields;
};

// IR Node that marks an free
struct FreeNode : public LowerIRNode {
    FreeNode(AbstractDataTypePtr data)
        : data(data) {
    }
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
    AbstractDataTypePtr parent;
    AbstractDataTypePtr child;
    std::vector<Expr> fields;
};

// IR Node marks a function call.
struct ComputeNode : public LowerIRNode {
    ComputeNode(FunctionCallPtr f, std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds)
        : f(f), new_ds(new_ds) {
    }
    FunctionCallPtr f;
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
};

// To track the interval nodes that need to be generated
// for the pipeline.
struct IntervalNode : public LowerIRNode {
public:
    IntervalNode(Variable v, Expr start, Expr end, Expr step, std::vector<LowerIR> body)
        : v(v), start(start), end(end), step(step), body(body) {
    }
    Variable v;
    Expr start;
    Expr end;
    Expr step;
    std::vector<LowerIR> body;
};

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct BlankNode : public LowerIRNode {
    BlankNode() = default;
};

}  // namespace gern

#endif