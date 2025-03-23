#include "codegen/optimizer.h"

namespace gern {

LowerIR Finalizer::finalize() {
    to_free.scope();

    visit(ir);

    std::vector<LowerIR> new_ir;
    new_ir.push_back(final_ir);

    auto free_ds = to_free.pop();
    for (auto ds : free_ds) {
        new_ir.push_back(new const FreeNode(ds));
    }

    return new const BlockNode(new_ir);
}

void Finalizer::visit(const AllocateNode *node) {
    if (node->data.freeAlloc()) {
        to_free.insert(node->data);
    }
    final_ir = new const AllocateNode(node->data,
                                      node->fields,
                                      node->f);
}

void Finalizer::visit(const FreeNode *node) {
    final_ir = new const FreeNode(node->data);
}

void Finalizer::visit(const InsertNode *node) {
    final_ir = new const InsertNode(node->parent, node->f);
}

void Finalizer::visit(const QueryNode *node) {
    if (node->child.freeQuery()) {
        to_free.insert(node->child);
    }
    final_ir = new const QueryNode(node->parent, node->child, node->fields, node->f);
}

void Finalizer::visit(const ComputeNode *node) {
    final_ir = new const ComputeNode(node->f, node->headers);
}

void Finalizer::visit(const IntervalNode *node) {
    std::vector<LowerIR> new_body;
    to_free.scope();

    visit(node->body);
    new_body.push_back(final_ir);

    auto free_ds = to_free.pop();
    for (auto ds : free_ds) {
        new_body.push_back(new const FreeNode(ds));
    }

    final_ir = new const IntervalNode(node->start,
                                      node->end,
                                      node->step,
                                      new const BlockNode(new_body),
                                      node->p);
}

void Finalizer::visit(const BlankNode *) {
    final_ir = new const BlankNode();
}

void Finalizer::visit(const DefNode *node) {
    final_ir = new const DefNode(node->assign, node->const_expr);
}

void Finalizer::visit(const AssertNode *node) {
    final_ir = new const AssertNode(node->constraint);
}

void Finalizer::visit(const BlockNode *node) {
    std::vector<LowerIR> new_ir;

    for (const auto &ir : node->ir_nodes) {
        visit(ir);
        new_ir.push_back(final_ir);
    }

    final_ir = new const BlockNode(new_ir);
}

void Finalizer::visit(const GridDeclNode *node) {
    final_ir = new const GridDeclNode(node->dim, node->v);
}

void Finalizer::visit(const SharedMemoryDeclNode *node) {
    final_ir = new const SharedMemoryDeclNode(node->size);
}

void Finalizer::visit(const OpaqueCall *node) {
    final_ir = new const OpaqueCall(node->f, node->headers);
}

util::ScopedSet<AbstractDataTypePtr> Finalizer::getToFree() const {
    return to_free;
}

}  // namespace gern
