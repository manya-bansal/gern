#pragma once

#include "compose/compose.h"
#include <iostream>

namespace gern {

class FunctionCall;
struct PipelineNode;

class CompositionVisitor {
public:
    virtual void visit(Compose);
    virtual void visit(Pipeline);
    virtual void visit(const FunctionCall *) = 0;
    virtual void visit(const PipelineNode *) = 0;
};

class ComposePrinter : public CompositionVisitor {
public:
    ComposePrinter(std::ostream &os, int ident)
        : os(os), ident(ident) {
    }
    void visit(Compose) override;
    void visit(Pipeline) override;
    virtual void visit(const FunctionCall *) override;
    virtual void visit(const PipelineNode *) override;

private:
    std::ostream &os;
    int ident = 0;
};

// Utility class to count how many distinct function calls
// a compose object contains.
class ComposeCounter : public CompositionVisitor {
public:
    ComposeCounter() = default;
    int numFuncs(Compose c);
    using CompositionVisitor::visit;
    virtual void visit(const FunctionCall *) override;
    virtual void visit(const PipelineNode *) override;

private:
    int num = 0;
};

class ComposeLower : public CompositionVisitor {
public:
};

}  // namespace gern