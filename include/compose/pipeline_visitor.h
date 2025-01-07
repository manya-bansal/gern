#pragma once

#include "compose/pipeline.h"
#include <iostream>

namespace gern {

class PipelineVisitor {
public:
    void visit(LowerIR);
    void visit(const Pipeline &);

    virtual void visit(const AllocateNode *) = 0;
    virtual void visit(const FreeNode *) = 0;
    virtual void visit(const InsertNode *) = 0;
    virtual void visit(const QueryNode *) = 0;
    virtual void visit(const ComputeNode *) = 0;
    virtual void visit(const IntervalNode *) = 0;
    virtual void visit(const DefNode *) = 0;
    virtual void visit(const BlankNode *) = 0;
};

class PipelinePrinter : PipelineVisitor {
public:
    PipelinePrinter(std::ostream &os, int ident)
        : os(os), ident(ident) {
    }
    using PipelineVisitor::visit;

    void visit(const Pipeline &);

    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const DefNode *);
    void visit(const BlankNode *);

private:
    std::ostream &os;
    int ident = 0;
};

}  // namespace gern