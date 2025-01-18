#pragma once

#include "annotations/abstract_function.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

using ComputeFunctionCallPtr = const ComputeFunctionCall *;
struct AllocateNode;
struct QueryNode;
struct InsertNode;
struct FreeNode;
struct ComputeNode;

class LowerIRVisitor;

struct LowerIRNode : public util::Manageable<LowerIRNode>,
                     public util::Uncopyable {
    LowerIRNode() = default;
    virtual ~LowerIRNode() = default;
    virtual void accept(LowerIRVisitor *) const = 0;
};

class LowerIR : public util::IntrusivePtr<const LowerIRNode> {
public:
    LowerIR()
        : util::IntrusivePtr<const LowerIRNode>(nullptr) {
    }
    LowerIR(const LowerIRNode *n)
        : util::IntrusivePtr<const LowerIRNode>(n) {
    }
    void accept(LowerIRVisitor *v) const;
};

std::ostream &operator<<(std::ostream &os, const LowerIR &n);

class ComposeLower : public CompositionVisitorStrict {
public:
    ComposeLower(Compose c)
        : c(c) {
    }

    using CompositionVisitorStrict::visit;
    LowerIR lower();
    void visit(const ComputeFunctionCall *c);
    void visit(const PipelineNode *c);

private:
    bool isIntermediate(AbstractDataTypePtr d) const;

    std::vector<LowerIR> generateAllDefs(const PipelineNode *node);       // Helper method to define all the variable definitions.
    std::vector<LowerIR> generateAllAllocs(const PipelineNode *node);     // Helper method to declare all the allocate node.
    std::vector<LowerIR> generateAllFrees(const PipelineNode *node);      // Helper method to declare all the frees.
    std::vector<LowerIR> generateAllQueries(const PipelineNode *parent);  // Helper method to declare all the frees.

    const QueryNode *constructQueryNode(AbstractDataTypePtr, std::vector<Expr>);     // Constructs a query node for a data-structure, and tracks this relationship.
    const FreeNode *constructFreeNode(AbstractDataTypePtr);                          // Constructs a free node for a data-structure, and tracks this relationship.
    const AllocateNode *constructAllocNode(AbstractDataTypePtr, std::vector<Expr>);  // Constructs a allocate for a data-structure, and tracks this relationship.
    std::vector<Compose> rewriteCalls(const PipelineNode *node) const;

    FunctionCall constructFunctionCall(FunctionSignature f, std::vector<Variable> ref_md_fields, std::vector<Expr> true_md_fields) const;  // Constructs a call with the true meta data fields mapped in the correct place.

    LowerIR generateConsumesIntervals(Pattern, std::vector<LowerIR> body) const;
    LowerIR generateOuterIntervals(Pattern, std::vector<LowerIR> body) const;

    std::vector<Assign> variable_definitions;
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;

    std::set<AbstractDataTypePtr> intermediates_set;  // All the intermediates visible to this pipeline.
    std::set<AbstractDataTypePtr> to_free;            // All the intermediates produces in reverse order.

    Compose c;
    LowerIR final_lower;
    bool has_been_lowered = false;
};

// IR Node that marks an allocation
struct AllocateNode : public LowerIRNode {
    AllocateNode(FunctionCall f)
        : f(f) {
    }
    void accept(LowerIRVisitor *) const;
    FunctionCall f;
};

// IR Node that marks an free
struct FreeNode : public LowerIRNode {
    FreeNode(AbstractDataTypePtr data)
        : data(data) {
    }
    void accept(LowerIRVisitor *) const;
    AbstractDataTypePtr data;
};

// IR Node that marks an insertion
// The child data structure is inserted
// into the parent data-structure as the
// subset with meta-data values in fields.
struct InsertNode : public LowerIRNode {
    InsertNode(AbstractDataTypePtr parent, FunctionCall f)
        : parent(parent), f(f) {
    }
    void accept(LowerIRVisitor *) const;
    AbstractDataTypePtr parent;
    FunctionCall f;
};

// IR Node that marks a query
// The child data structure is produced
// from the parent data-structure corresponding to
// the subset with meta-data values in fields.
struct QueryNode : public LowerIRNode {
    QueryNode(AbstractDataTypePtr parent, FunctionCall f)
        : parent(parent), f(f) {
    }
    void accept(LowerIRVisitor *) const;
    AbstractDataTypePtr parent;
    FunctionCall f;
};

// IR Node marks a FunctionSignature call.
struct ComputeNode : public LowerIRNode {
    ComputeNode(FunctionCall f,
                std::vector<std::string> headers)
        : f(f), headers(headers) {
    }
    void accept(LowerIRVisitor *) const;
    FunctionCall f;
    std::vector<std::string> headers;
};

// Block Nodes can hold a list of IR nodes.
struct BlockNode : public LowerIRNode {
    BlockNode(std::vector<LowerIR> ir_nodes)
        : ir_nodes(ir_nodes) {
    }
    void accept(LowerIRVisitor *) const;
    std::vector<LowerIR> ir_nodes;
};

// To track the interval nodes that need to be generated
// for the pipeline.
struct IntervalNode : public LowerIRNode {
public:
    IntervalNode(Assign start, Expr end, Expr step, LowerIR body)
        : start(start), end(end), step(step), body(body) {
    }
    void accept(LowerIRVisitor *) const;

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
    LowerIR body;
};

// Node to declare definitions of variables.
struct DefNode : public LowerIRNode {
    DefNode(Assign assign, bool const_expr)
        : assign(assign), const_expr(const_expr) {
    }

    void accept(LowerIRVisitor *) const;
    Assign assign;
    bool const_expr;  // Track whether this is a actually a constexpr definition.
};

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct BlankNode : public LowerIRNode {
    BlankNode() = default;
    void accept(LowerIRVisitor *) const;
};

// Function boundary indicates that the corresponding
// lowered nodes are called in a separate function body.
// These may, or may not be, fused with the rest of the code.
struct FunctionBoundary : public LowerIRNode {
    FunctionBoundary(LowerIR nodes)
        : nodes(nodes) {
    }
    void accept(LowerIRVisitor *) const;
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> queried_names;
    LowerIR nodes;
};

}  // namespace gern