#pragma once

#include "annotations/data_dependency_language.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

class PipelineVisitor;
struct PipelineNode;
struct AllocateNode;
struct QueryNode;
struct InsertNode;
struct FreeNode;
struct ComputeNode;

using ComputeFunctionCallPtr = const ComputeFunctionCall *;

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

struct FunctionCall {
    std::string name;
    std::vector<Expr> args;
    std::vector<Expr> template_args;
    Argument output = Argument();
};

// The pipeline actually holds the lowered
// nodes, and helps us nest pipelines.
class Pipeline : public CompositionVisitorStrict {

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

    using CompositionVisitorStrict::visit;
    void visit(const ComputeFunctionCall *c);
    void visit(const PipelineNode *c);

    std::vector<LowerIR> getIRNodes() const;
    // Returns the FunctionSignature call that produces a particular output.
    std::set<AbstractDataTypePtr> getInputs() const;
    AbstractDataTypePtr getOutput() const;
    std::set<ComputeFunctionCallPtr> getConsumerFunctions(AbstractDataTypePtr) const;

    void accept(CompositionVisitorStrict *) const;

    std::set<AbstractDataTypePtr> getAllWriteDataStruct() const;  // This gathers all the data-structures written to in the pipeline.
    std::set<AbstractDataTypePtr> getAllReadDataStruct() const;   // This gathers all the data-structures written to in the pipeline.
private:
    ComputeFunctionCallPtr getProducerFunction(AbstractDataTypePtr ds) const;
    void init(std::vector<Compose> compose);  // Initializes private vars, and ensures that the user has constructed a valid pipeline.
    bool isIntermediate(AbstractDataTypePtr d) const;

    void generateAllDefs();    // Helper method to define all the variable definitions.
    void generateAllAllocs();  // Helper method to declare all the allocate node.
    void generateAllFrees();   // Helper method to declare all the frees.

    const QueryNode *constructQueryNode(AbstractDataTypePtr, std::vector<Expr>);         // Constructs a query node for a data-structure, and tracks this relationship.
    const FreeNode *constructFreeNode(AbstractDataTypePtr);                              // Constructs a free node for a data-structure, and tracks this relationship.
    const AllocateNode *constructAllocNode(AbstractDataTypePtr, std::vector<Variable>);  // Constructs a allocate for a data-structure, and tracks this relationship.

    FunctionSignature constructFunction(FunctionSignature f, std::vector<Variable> ref_md_fields, std::vector<Variable> true_md_fields) const;  // Constructs a call with the true meta data fields mapped in the correct place.
    FunctionCall constructFunctionCall(FunctionSignature f, std::vector<Variable> ref_md_fields, std::vector<Expr> true_md_fields) const;       // Constructs a call with the true meta data fields mapped in the correct place.

    std::vector<LowerIR> generateConsumesIntervals(ComputeFunctionCallPtr, std::vector<LowerIR> body) const;
    std::vector<LowerIR> generateOuterIntervals(ComputeFunctionCallPtr, std::vector<LowerIR> body) const;

    std::vector<Compose> compose;
    std::vector<LowerIR> lowered;
    std::vector<Assign> variable_definitions;
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;

    std::set<AbstractDataTypePtr> intermediates;        // All the intermediates visible to this pipeline.
    std::vector<AbstractDataTypePtr> outputs_in_order;  // All the intermediates produces in reverse order.
    std::set<AbstractDataTypePtr> to_free;              // All the intermediates produces in reverse order.
    AbstractDataTypePtr true_output;                    // The output that this pipeline generates.

    bool has_been_lowered = false;
    bool device = false;
};

std::ostream &operator<<(std::ostream &os, const Pipeline &p);

// IR Node that marks an allocation
struct AllocateNode : public LowerIRNode {
    AllocateNode(FunctionSignature f)
        : f(f) {
    }
    void accept(PipelineVisitor *) const;
    FunctionSignature f;
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
    InsertNode(AbstractDataTypePtr parent, FunctionSignature f)
        : parent(parent), f(f) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr parent;
    FunctionSignature f;
};

// IR Node that marks a query
// The child data structure is produced
// from the parent data-structure corresponding to
// the subset with meta-data values in fields.
struct QueryNode : public LowerIRNode {
    QueryNode(AbstractDataTypePtr parent, FunctionCall f)
        : parent(parent), f(f) {
    }
    void accept(PipelineVisitor *) const;
    AbstractDataTypePtr parent;
    FunctionCall f;
};

// IR Node marks a FunctionSignature call.
struct ComputeNode : public LowerIRNode {
    ComputeNode(FunctionSignature f,
                std::vector<std::string> headers)
        : f(f), headers(headers) {
    }
    void accept(PipelineVisitor *) const;
    FunctionSignature f;
    std::vector<std::string> headers;
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

    void accept(CompositionVisitorStrict *) const;
    Pipeline p;
};

// Node to declare definitions of variables.
struct DefNode : public LowerIRNode {
    DefNode(Assign assign, bool const_expr)
        : assign(assign), const_expr(const_expr) {
    }

    void accept(PipelineVisitor *) const;
    Assign assign;
    bool const_expr;  // Track whether this is a actually a constexpr definition.
};

// Filler Node to manipulate objects (like vectors, etc)
// while iterating over them.
struct BlankNode : public LowerIRNode {
    BlankNode() = default;
    void accept(PipelineVisitor *) const;
};

// Defining an abstract data class that we can use to define query and free node.
class PipelineDS : public AbstractDataType {
public:
    PipelineDS(const std::string &name,
               const std::string &type,
               const std::vector<Variable> &fields,
               const FunctionSignature &allocate,
               const FunctionSignature &free,
               const FunctionSignature &insert,
               const FunctionSignature &query,
               const bool &to_free)
        : name(name), type(type), fields(fields),
          allocate(allocate), free(free),
          insert(insert), query(query),
          to_free(to_free) {
    }

    virtual std::string getName() const override {
        return name;
    }

    virtual std::string getType() const override {
        return type;
    }

    std::vector<Variable> getFields() const override {
        return fields;
    }
    FunctionSignature getAllocateFunction() const override {
        return allocate;
    }
    FunctionSignature getFreeFunction() const override {
        return free;
    }
    FunctionSignature getInsertFunction() const override {
        return insert;
    }
    FunctionSignature getQueryFunction() const override {
        return query;
    }

    // Tracks whether any of the queries need to be free,
    // or if they are actually returning views.
    bool freeQuery() const override {
        return to_free;
    }

    static AbstractDataTypePtr make(const std::string &name, AbstractDataTypePtr ds) {
        return AbstractDataTypePtr(new const PipelineDS(name, ds.ptr->getType(),
                                                        ds.ptr->getFields(),
                                                        ds.ptr->getAllocateFunction(),
                                                        ds.ptr->getFreeFunction(),
                                                        ds.ptr->getInsertFunction(),
                                                        ds.ptr->getQueryFunction(),
                                                        ds.ptr->freeQuery()));
    }

private:
    std::string name;
    std::string type;
    std::vector<Variable> fields;
    FunctionSignature allocate;
    FunctionSignature free;
    FunctionSignature insert;
    FunctionSignature query;
    bool to_free;
};

}  // namespace gern