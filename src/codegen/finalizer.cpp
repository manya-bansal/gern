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
    ir = scoper.construct();

    to_free.scope();

    visit(ir);

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

LowerIR ADTReuser::construct() {
    cur_lno = 0;
    visit(ir);

    // Now that we have the live ranges, loop over all the allocate calls
    // and figure out which data-structures can be reused.
    std::map<AbstractDataTypePtr, std::set<AbstractDataTypePtr>> reusable_adts;
    // for (const auto &adt : allocate_calls) {
    //     // Look into the map and see if there is an adt that this adt does not conflict with.
    //     FunctionCall call = adt.second;
    //     for (auto &reusable_set : reusable_adts) {

    //         bool can_reuse = true;
    //         for (auto &reusable_adt : reusable_set) {
    //             if (std::get<1>(live_range[reusable_adt]) < std::get<0>(live_range[adt.first])) {
    //                 can_reuse = false;
    //                 break;
    //             }
    //             reusable_set.insert(adt.first);
    //             break;
    //         }
    //     }
    // }

    return ir;
}

void ADTReuser::visit(const AllocateNode *node) {
    cur_lno++;
    // The subset object becomes live at the point of allocation.
    live_range[to<DSArg>(node->f.output)->getADTPtr()] = std::make_tuple(cur_lno, cur_lno);
    allocate_calls[to<DSArg>(node->f.output)->getADTPtr()] = node->f;
}

void ADTReuser::visit(const FreeNode *node) {
    cur_lno++;
    update_live_range(node->call.data);
}

void ADTReuser::visit(const InsertNode *node) {
    cur_lno++;
    // Update parent adt live range.
    update_live_range(node->call.data);
    // Update child adt live range.
    std::vector<Argument> args = node->call.call.args;
    AbstractDataTypePtr child_adt = to<DSArg>(args[args.size() - 1])->getADTPtr();
    update_live_range(child_adt);
}

void ADTReuser::visit(const QueryNode *node) {
    cur_lno++;
    // Update parent adt live range.
    update_live_range(node->parent);
    // Child appears for the first time.
    live_range[node->child] = std::make_tuple(cur_lno, cur_lno);
}

void ADTReuser::visit(const ComputeNode *node) {
    cur_lno++;
    // Update adt live range.
    update_live_range(node->adt);
    // Update all the arguments.
    for (const auto &arg : node->f.args) {
        if (isa<DSArg>(arg)) {
            update_live_range(to<DSArg>(arg)->getADTPtr());
        }
    }
}

void ADTReuser::visit(const IntervalNode *node) {
    cur_lno++;
    // Update the body.
    visit(node->body);
}

void ADTReuser::visit(const BlankNode *) {
}

void ADTReuser::visit(const DefNode *node) {
    update_live_range(node->assign.getA());
    update_live_range(node->assign.getB());
}

void ADTReuser::visit(const AssertNode *node) {
    update_live_range(node->constraint.getA());
    update_live_range(node->constraint.getB());
}

void ADTReuser::visit(const BlockNode *node) {
    for (const auto &ir : node->ir_nodes) {
        visit(ir);
    }
}

void ADTReuser::visit(const GridDeclNode *) {
    cur_lno++;
}

void ADTReuser::visit(const SharedMemoryDeclNode *) {
}

void ADTReuser::visit(const OpaqueCall *node) {
    // Update all the arguments.
    for (const auto &arg : node->f.args) {
        if (isa<DSArg>(arg)) {
            update_live_range(to<DSArg>(arg)->getADTPtr());
        }
    }
}

void ADTReuser::update_live_range(AbstractDataTypePtr adt) {
    if (!live_range.contains(adt)) {
        throw error::InternalError("Free of undeclared subset object: " + adt.str());
    }
    int first_use = std::get<0>(live_range[adt]);
    live_range[adt] = std::make_tuple(first_use, cur_lno);
}

void ADTReuser::update_live_range(Expr e) {
    match(e,
          std::function<void(const ADTMemberNode *)>(
              [&](const ADTMemberNode *op) {
                  update_live_range(op->ds);
              }));
}

LowerIR Scoper::construct() {
    visit(ir);
    std::vector<LowerIR> new_body = new_statements[cur_scope];
    LowerIR final_ir = new const BlockNode(new_body);
    return final_ir;
}

int32_t Scoper::get_scope(Expr e) const {
    int32_t scope = 0;
    match(e,
          std::function<void(const VariableNode *)>(
              [&](const VariableNode *op) {
                  scope = std::max(scope, var_scope.contains(op) ? var_scope.at(op) : 0);
              }),
          std::function<void(const ADTMemberNode *)>(
              [&](const ADTMemberNode *op) {
                  scope = std::max(scope, adt_scope.contains(op->ds) ? adt_scope.at(op->ds) : 0);
              }),
          std::function<void(const GridDimNode *)>(
              [&](const GridDimNode *) {
                  scope = std::max(scope, cur_scope);
              }));
    return scope;
}

int32_t Scoper::get_scope(std::vector<Argument> args) const {
    int32_t scope = 0;
    for (const auto &arg : args) {
        if (isa<DSArg>(arg)) {
            auto adt = to<DSArg>(arg)->getADTPtr();
            scope = std::max(scope, adt_scope.contains(adt) ? adt_scope.at(adt) : 0);
        } else if (isa<ExprArg>(arg)) {
            scope = std::max(scope, get_scope(to<ExprArg>(arg)->getExpr()));
        } else {
            throw error::InternalError("Unknown argument type: " + arg.str());
        }
    }
    return std::min(scope, cur_scope);
}

void Scoper::visit(const AllocateNode *node) {
    // Scope is the maximum scope of the arguments.
    adt_scope[to<DSArg>(node->f.output)->getADTPtr()] = get_scope(node->f.args);
    new_statements[adt_scope[to<DSArg>(node->f.output)->getADTPtr()]].push_back(node);
}

void Scoper::visit(const FreeNode *) {
    throw error::InternalError("No Frees at this point!");
}

void Scoper::visit(const InsertNode *node) {
    adt_scope[node->call.data] = get_scope(node->call.call.args);
    new_statements[adt_scope[node->call.data]].push_back(node);
}

void Scoper::visit(const QueryNode *node) {
    std::vector<Argument> scoped_by = node->call.call.args;
    scoped_by.push_back(Argument(node->parent));
    adt_scope[node->child] = get_scope(scoped_by);
    new_statements[adt_scope[node->child]].push_back(node);
}

void Scoper::visit(const ComputeNode *node) {
    adt_scope[node->adt] = get_scope(node->f.args);
    new_statements[adt_scope[node->adt]].push_back(node);
}

void Scoper::visit(const IntervalNode *node) {
    cur_scope++;
    var_scope[node->getIntervalVariable()] = cur_scope;
    visit(node->body);

    // Generate the new body!
    std::vector<LowerIR> new_body = new_statements[cur_scope];
    new_statements.erase(cur_scope);
    cur_scope--;

    // Insert the new interval node with the new body at the previous scope.
    new_statements[cur_scope].push_back(new const IntervalNode(node->start,
                                                               node->end,
                                                               node->step,
                                                               new const BlockNode(new_body),
                                                               node->p));
}

void Scoper::visit(const DefNode *node) {
    int32_t scope = get_scope(node->assign.getB());
    var_scope[to<Variable>(node->assign.getA())] = scope;
    new_statements[scope].push_back(node);
}

void Scoper::visit(const AssertNode *node) {
    int32_t scope = get_scope(node->constraint.getA());
    int32_t scope_b = get_scope(node->constraint.getB());
    new_statements[std::max(scope, scope_b)].push_back(node);
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
