#pragma once

#include "codegen/lower.h"
#include "codegen/lower_visitor.h"
#include "utils/scoped_set.h"

namespace gern {

class Finalizer : public LowerIRVisitor {
public:
    Finalizer(LowerIR ir)
        : ir(ir) {
    }

    LowerIR finalize();
    using LowerIRVisitor::visit;
    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    // void visit(const FunctionBoundary *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

    util::ScopedSet<AbstractDataTypePtr> getToFree() const;

private:
    LowerIR ir;
    LowerIR final_ir;
    util::ScopedSet<AbstractDataTypePtr> to_free;
    util::ScopedSet<const InsertNode *> to_insert;
};

class Scoper : public LowerIRVisitor {
public:
    Scoper(LowerIR ir)
        : ir(ir) {
    }

    // visit each node, and figure out the scope of each node.
    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    // void visit(const FunctionBoundary *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

    /**
    * Hoist allocations of temporary data-structures as much as possible.
    */
    LowerIR hoist_allocations(LowerIR ir);

private:
    void init_scope();
    /**
     * Tracks the scope of each IR node.
     */
    std::map<LowerIRNode *, Variable> scope;
    /**
     * Maps the scope variable to the interval that introduces it.
     */
    std::map<Variable, IntervalNode *> scope_to_interval;
    LowerIR ir;
};

}  // namespace gern
