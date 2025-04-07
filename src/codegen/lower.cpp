#include "codegen/lower.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "codegen/concretize.h"
#include "codegen/lower_visitor.h"
#include "utils/name_generator.h"

namespace gern {

void LowerIR::accept(LowerIRVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const LowerIR &n) {
    LowerPrinter print(os, 0);
    print.visit(n);
    return os;
}

void AllocateNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void FreeNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void InsertNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void GridDeclNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void SharedMemoryDeclNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void QueryNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void ComputeNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void IntervalNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void DefNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void AssertNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

bool IntervalNode::isMappedToGrid() const {
    return p != Grid::Unit::UNDEFINED;
}

Variable IntervalNode::getIntervalVariable() const {
    std::set<Variable> v = getVariables(start.getA());
    return *(v.begin());
}

void BlankNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

// void FunctionBoundary::accept(LowerIRVisitor *v) const {
//     v->visit(this);
// }

void OpaqueCall::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

void BlockNode::accept(LowerIRVisitor *v) const {
    v->visit(this);
}

}  // namespace gern