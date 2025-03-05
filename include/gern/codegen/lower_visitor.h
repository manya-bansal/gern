#pragma once

#include "codegen/lower.h"
#include <iostream>

namespace gern {

class LowerIRVisitor {
public:
    void visit(LowerIR);

    virtual void visit(const AllocateNode *) = 0;
    virtual void visit(const FreeNode *) = 0;
    virtual void visit(const InsertNode *) = 0;
    virtual void visit(const QueryNode *) = 0;
    virtual void visit(const ComputeNode *) = 0;
    virtual void visit(const IntervalNode *) = 0;
    virtual void visit(const DefNode *) = 0;
    virtual void visit(const AssertNode *) = 0;
    virtual void visit(const BlankNode *) = 0;
    virtual void visit(const FunctionBoundary *) = 0;
    virtual void visit(const BlockNode *) = 0;
    virtual void visit(const GridDeclNode *) = 0;
    virtual void visit(const SharedMemoryDeclNode *) = 0;
    virtual void visit(const OpaqueCall *) = 0;
};

class LowerPrinter : LowerIRVisitor {
public:
    LowerPrinter(std::ostream &os, int ident)
        : os(os), ident(ident) {
    }
    using LowerIRVisitor::visit;

    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const DefNode *);
    void visit(const BlankNode *);
    void visit(const AssertNode *);
    void visit(const FunctionBoundary *);
    void visit(const BlockNode *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);
private:
    std::ostream &os;
    int ident = 0;
};

}  // namespace gern