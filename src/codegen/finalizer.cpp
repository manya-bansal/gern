#include "codegen/finalizer.h"
#include "annotations/rewriter.h"
#include "math.h"

namespace gern {

static MethodCall makeFreeCall(AbstractDataTypePtr ds) {
    FunctionCall free{
        .name = ds.getFreeFunction().name,
        .args = {},
        .template_args = {},
        .grid = LaunchArguments(),
        .block = LaunchArguments(),
        .access = Access::HOST,
        .smem_size = Expr(),
    };
    return MethodCall(ds, free);
}

LowerIR Finalizer::finalize() {

    Scoper scoper(ir);
    LowerIR hoisted_ir = scoper.construct();

    to_free.scope();

    visit(hoisted_ir);

    std::vector<LowerIR> new_ir;
    new_ir.push_back(final_ir);

    auto free_ds = to_free.pop();
    for (auto ds : free_ds) {
        new_ir.push_back(new const FreeNode(makeFreeCall(ds)));
    }

    return new const BlockNode(new_ir);
}

void Finalizer::visit(const AllocateNode *node) {
    auto output = node->f.output;
    assert(isa<DSArg>(output));
    auto adt = to<DSArg>(output)->getADTPtr();
    if (adt.freeAlloc()) {
        to_free.insert(adt);
    }
    final_ir = new const AllocateNode(node->f);
}

void Finalizer::visit(const FreeNode *node) {
    final_ir = new const FreeNode(node->call);
}

void Finalizer::visit(const InsertNode *node) {
    final_ir = new const InsertNode(node->call);
}

void Finalizer::visit(const QueryNode *node) {
    if (node->child.freeQuery()) {
        to_free.insert(node->child);
    }
    final_ir = new const QueryNode(node->parent, node->child, node->fields, node->call);
}

void Finalizer::visit(const ComputeNode *node) {
    final_ir = new const ComputeNode(node->f, node->headers, node->adt);
}

void Finalizer::visit(const IntervalNode *node) {
    std::vector<LowerIR> new_body;
    to_free.scope();

    visit(node->body);
    new_body.push_back(final_ir);

    auto free_ds = to_free.pop();
    for (auto ds : free_ds) {
        new_body.push_back(new const FreeNode(makeFreeCall(ds)));
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

LowerIR Scoper::construct() {
    visit(ir);
    std::vector<LowerIR> new_body = new_statements[cur_scope];
    LowerIR final_ir = new const BlockNode(new_body);
    std::cout << "FinalIR: " << final_ir << std::endl;
    return final_ir;
}

int32_t Scoper::get_scope(Expr e) const {
    int32_t scope = 0;
    match(e, std::function<void(const VariableNode *)>(
                 [&](const VariableNode *op) {
                     // Get the most nested scope.
                     scope = std::max(scope, get_scope_var(op));
                 }));
    return scope;
}

int32_t Scoper::get_scope(std::vector<Argument> args) const {
    int32_t scope = 0;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg)) {
            auto adt = to<DSArg>(arg)->getADTPtr();
            scope = std::max(scope, adt_scope.contains(adt) ? adt_scope.at(adt) : 0);
        }
        if (isa<VarArg>(arg)) {
            scope = std::max(scope, get_scope_var(to<VarArg>(arg)->getVar()));
        }
        if (isa<ExprArg>(arg)) {
            scope = std::max(scope, get_scope(to<ExprArg>(arg)->getExpr()));
        }
        // } else {

        //     throw error::InternalError("Unknown argument type: " + arg.str());
        // }
    }
    return scope;
}

void Scoper::visit(const AllocateNode *node) {
    // Loop through the arguments and get the scope.
    adt_scope[to<DSArg>(node->f.output)->getADTPtr()] = get_scope(node->f.args);
    new_statements[adt_scope[to<DSArg>(node->f.output)->getADTPtr()]].push_back(node);
    std::cout << "AllocateNode: " << node->f.output.str() << " " << adt_scope[to<DSArg>(node->f.output)->getADTPtr()] << std::endl;
}

void Scoper::visit(const FreeNode *) {
    throw error::InternalError("No Frees at this point!");
}

void Scoper::visit(const InsertNode *node) {
    adt_scope[node->call.data] = get_scope(node->call.call.args);
    std::cout << "InsertNode: " << node->call.data.str() << " " << adt_scope[node->call.data] << std::endl;
    new_statements[adt_scope[node->call.data]].push_back(node);
}

void Scoper::visit(const QueryNode *node) {
    std::vector<Argument> scoped_by = node->call.call.args;
    scoped_by.push_back(Argument(node->parent));
    adt_scope[node->child] = get_scope(scoped_by);
    new_statements[adt_scope[node->child]].push_back(node);
    std::cout << "QueryNode: " << node->child.str() << " " << adt_scope[node->child] << std::endl;
}

void Scoper::visit(const ComputeNode *node) {
    adt_scope[node->adt] = get_scope(node->f.args);
    new_statements[adt_scope[node->adt]].push_back(node);
    std::cout << "ComputeNode: " << node->adt.str() << " " << adt_scope[node->adt] << std::endl;
}

void Scoper::visit(const IntervalNode *node) {
    cur_scope++;
    var_scope[node->getIntervalVariable()] = cur_scope;
    var_stack.push_back(node->getIntervalVariable());
    visit(node->body);

    // Generate the new body!
    std::vector<LowerIR> new_body = new_statements[cur_scope];
    new_statements.erase(cur_scope);
    cur_scope--;

    // Insert the new interval node!.
    new_statements[cur_scope].push_back(new const IntervalNode(node->start,
                                                               node->end,
                                                               node->step,
                                                               new const BlockNode(new_body),
                                                               node->p));
}

int32_t Scoper::get_scope_var(Variable v) const {
    if (!var_scope.contains(v)) {
        return 0;
    }
    return var_scope.at(v);
}

void Scoper::visit(const DefNode *node) {
    int32_t scope = 0;
    match(node->assign.getB(), std::function<void(const VariableNode *)>(
                                   [&](const VariableNode *op) {
                                       // Get the most nested scope.
                                       scope = std::max(scope, get_scope(op));
                                   }));
    var_scope[to<Variable>(node->assign.getA())] = scope;
    new_statements[scope].push_back(node);
    std::cout << "DefNode: " << node->assign.getA().str() << " " << scope << std::endl;
}

void Scoper::visit(const AssertNode *node) {
    new_statements[cur_scope].push_back(node);
}

void Scoper::visit(const BlockNode *node) {
    for (const auto &ir : node->ir_nodes) {
        visit(ir);
    }
}

void Scoper::visit(const GridDeclNode *node) {
    new_statements[cur_scope].push_back(node);
}

void Scoper::visit(const SharedMemoryDeclNode *node) {
    new_statements[cur_scope].push_back(node);
}

void Scoper::visit(const OpaqueCall *node) {
    new_statements[cur_scope].push_back(node);
}

void Scoper::visit(const BlankNode *) {
}

}  // namespace gern
