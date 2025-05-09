#pragma once

#include "annotations/abstract_function.h"
#include "compose/composable.h"
#include "compose/composable_visitor.h"
#include "compose/compose.h"
#include "utils/scoped_map.h"
#include "utils/scoped_set.h"
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
    FreeNode(MethodCall call)
        : call(call) {
    }
    void accept(LowerIRVisitor *) const;
    MethodCall call;
};

// IR Node that marks an insertion
// The child data structure is inserted
// into the parent data-structure as the
// subset with meta-data values in fields.
struct InsertNode : public LowerIRNode {
    InsertNode(MethodCall call)
        : call(call) {
    }
    void accept(LowerIRVisitor *) const;
    MethodCall call;
};

struct GridDeclNode : public LowerIRNode {
    GridDeclNode(const Grid::Dim &dim, Variable v)
        : dim(dim), v(v) {
    }
    void accept(LowerIRVisitor *) const;
    Grid::Dim dim;
    Variable v;
};

// struct GlobalDeclNode : public LowerIRNode {
//     GlobalDeclNode(std::map<Grid::Dim, Variable> grid_dims,
//                    Variable shared_mem,
//                    LowerIR body);

//     void accept(LowerIRVisitor *) const;
//     std::map<Grid::Dim, Variable> grid_dims;
//     Variable shared_mem;
//     LowerIR body;
// };

struct SharedMemoryDeclNode : public LowerIRNode {
    SharedMemoryDeclNode(Variable size)
        : size(size) {
    }
    void accept(LowerIRVisitor *) const;
    Variable size;
};

// IR Node that marks a query
// The child data structure is produced
// from the parent data-structure corresponding to
// the subset with meta-data values in fields.
struct QueryNode : public LowerIRNode {
    QueryNode(AbstractDataTypePtr parent,
              AbstractDataTypePtr child,
              const std::vector<Expr> &fields,
              MethodCall call,
              const bool &insert,
              LowerIR insert_call)
        : parent(parent), child(child), fields(fields), call(call), insert(insert), insert_call(insert_call) {
    }
    void accept(LowerIRVisitor *) const;
    AbstractDataTypePtr parent;
    AbstractDataTypePtr child;
    std::vector<Expr> fields;
    MethodCall call;
    bool insert;
    LowerIR insert_call;
};

// IR Node marks a FunctionSignature call.
struct ComputeNode : public LowerIRNode {
    ComputeNode(FunctionCall f,
                std::vector<std::string> headers,
                AbstractDataTypePtr adt)
        : f(f), headers(headers), adt(adt) {
    }
    void accept(LowerIRVisitor *) const;
    FunctionCall f;
    std::vector<std::string> headers;
    AbstractDataTypePtr adt;
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
    IntervalNode(Assign start, Expr end, Expr step, LowerIR body, Grid::Unit p)
        : start(start), end(end), step(step), body(body), p(p) {
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
    Grid::Unit p;
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

// Node to declare definitions of variables.
struct AssertNode : public LowerIRNode {
    AssertNode(Constraint constraint)
        : constraint(constraint) {
    }

    void accept(LowerIRVisitor *) const;
    Constraint constraint;
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
// struct FunctionBoundary : public LowerIRNode {
//     FunctionBoundary(LowerIR nodes)
//         : nodes(nodes) {
//     }
//     void accept(LowerIRVisitor *) const;
//     std::map<AbstractDataTypePtr, AbstractDataTypePtr> queried_names;
//     LowerIR nodes;
// };

struct OpaqueCall : public LowerIRNode {
    OpaqueCall(FunctionCall f,
               std::vector<std::string> headers)
        : f(f), headers(headers) {
    }
    void accept(LowerIRVisitor *) const;
    FunctionCall f;
    std::vector<std::string> headers;
};

// Defining an abstract data class that we can use to define query and free node.
class DummyDS : public AbstractDataType {
public:
    DummyDS(const std::string &name,
            const std::string &type,
            const std::vector<Variable> &fields,
            const FunctionSignature &allocate,
            const FunctionSignature &free,
            const FunctionSignature &insert,
            const FunctionSignature &query,
            const bool &to_free,
            const bool &insert_query)
        : name(name), type(type), fields(fields),
          allocate(allocate), free(free),
          insert(insert), query(query),
          to_free(to_free), insert_query(insert_query) {
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

    bool insertQuery() const override {
        return insert_query;
    }

    static AbstractDataTypePtr make(const std::string &name,
                                    const std::string &type,
                                    AbstractDataTypePtr ds) {
        return AbstractDataTypePtr(new const DummyDS(name, type,
                                                     ds.ptr->getFields(),
                                                     ds.ptr->getAllocateFunction(),
                                                     ds.ptr->getFreeFunction(),
                                                     ds.ptr->getInsertFunction(),
                                                     ds.ptr->getQueryFunction(),
                                                     ds.ptr->freeQuery(),
                                                     ds.ptr->insertQuery()));
    }

    static AbstractDataTypePtr make(const std::string &name,
                                    AbstractDataTypePtr ds) {
        return make(name, ds.getType(), ds);
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
    bool insert_query;
};

}  // namespace gern