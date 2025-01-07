#include "codegen/codegen.h"
#include "annotations/data_dependency_language.h"
#include "annotations/grid.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

CGStmt CodeGenerator::generate_code(const Pipeline &p) {
    // Lower each IR node one by one.
    std::vector<CGStmt> lowered_nodes;
    for (const auto &node : p.getIRNodes()) {
        this->visit(node);
        lowered_nodes.push_back(code);
    }
    code = Block::make(lowered_nodes);

    // Once we have visited the pipeline, we need to
    // wrap the body in a function interface.

    // Add all the data-structures that haven't been declared
    // or allocated to the function arguments. These are the
    // true inputs, or the true outputs.
    std::vector<AbstractDataTypePtr> to_declare_adts;
    std::set_difference(used_adt.begin(), used_adt.end(),
                        declared_adt.begin(), declared_adt.end(),
                        std::back_inserter(to_declare_adts));

    std::vector<Variable> to_declare_vars;
    std::set_difference(used.begin(), used.end(),
                        declared.begin(), declared.end(),
                        std::back_inserter(to_declare_vars),
                        std::less<Variable>());

    std::vector<CGExpr> args;
    std::vector<CGStmt> hook_body;
    std::vector<CGExpr> hook_args;

    int num_args = 0;
    for (const auto &ds : to_declare_adts) {
        args.push_back(VarDecl::make(Type::make(ds->getType()), ds->getName()));
        hook_body.push_back(
            VarAssign::make(VarDecl::make(Type::make(ds->getType()), ds->getName()),
                            Deref::make(
                                Cast::make(Type::make(ds->getType() + "*"),
                                           Var::make("args[" + std::to_string(num_args) + "]")))));
        hook_args.push_back(Var::make(ds->getName()));
        argument_order.push_back(ds->getName());
        num_args++;
    }

    for (const auto &v : to_declare_vars) {
        args.push_back(VarDecl::make(Type::make(v.getType().str()), v.getName()));
        hook_body.push_back(
            VarAssign::make(VarDecl::make(Type::make(v.getType().str()), v.getName()),
                            Deref::make(
                                Cast::make(Type::make(v.getType().str() + "*"),
                                           Var::make("args[" + std::to_string(num_args) + "]")))));
        hook_args.push_back(gen(v));
        argument_order.push_back(v.getName());
        num_args++;
    }

    // The return type is always void, the output
    // is modified by reference.
    code = DeclFunc::make(name, Type::make("void"), args, code, p.is_at_device());

    // Also make the function that is used as the entry point for
    // user code.
    std::vector<CGStmt> full_code;
    CGStmt hook_call = VoidCall::make(Call::make(name, hook_args));
    if (p.is_at_device()) {
        // Declare the grid and block dimensions.
        hook_body.push_back(EscapeCGStmt::make("dim3 __grid_dim__(" + grid_dim.str() + ");"));
        hook_body.push_back(EscapeCGStmt::make("dim3 __block_dim__(" + block_dim.str() + ");"));
        hook_call = KernelLaunch::make(name, hook_args, Var::make("__grid_dim__"), Var::make("__block_dim__"));
        full_code.push_back(EscapeCGStmt::make("#include <cuda_runtime.h>"));
    }

    hook_body.push_back(hook_call);

    CGStmt hook = DeclFunc::make(hook_name, Type::make("void"),
                                 {VarDecl::make(Type::make("void**"), "args")},
                                 Block::make(hook_body));

    // Now, add the headers.
    for (const auto &h : headers) {
        full_code.push_back(EscapeCGStmt::make("#include \"" + h + "\""));
    }

    full_code.push_back(BlankLine::make());
    full_code.push_back(BlankLine::make());
    full_code.push_back(EscapeCGStmt::make("extern \"C\""));
    full_code.push_back(Scope::make(Block::make({code, hook})));

    return Block::make(full_code);
}

void CodeGenerator::visit(const AllocateNode *op) {
    std::string method_call = op->data->getType() + "::allocate";
    std::vector<CGExpr> args;
    for (const auto &a : op->fields) {
        args.push_back(gen(a));
    }
    CGExpr lhs = VarDecl::make(Type::make(op->data->getType()),
                               op->data->getName());
    code = VarAssign::make(lhs, Call::make(method_call, args));

    declared_adt.insert(op->data);
}

void CodeGenerator::visit(const FreeNode *op) {

    DEBUG(
        if (declared_adt.count(op->data) > 0) {
            throw error::InternalError("Freeing a data-structure that hasn't been allocated??");
        })

    std::string method_call = op->data->getName() + ".destroy";
    code = VoidCall::make(Call::make(method_call, {}));
}

void CodeGenerator::visit(const InsertNode *op) {
    std::string method_call = op->parent->getName() + ".insert";
    std::vector<CGExpr> args;
    for (const auto &a : op->fields) {
        args.push_back(gen(a));
    }
    CGExpr lhs = VarDecl::make(Type::make(op->child->getType()),
                               op->child->getName());
    code = VarAssign::make(lhs, Call::make(method_call, args));

    declared_adt.insert(op->child);
    used_adt.insert(op->parent);
}

void CodeGenerator::visit(const QueryNode *op) {
    std::string method_call = op->parent->getName() + ".query";
    std::vector<CGExpr> args;
    for (const auto &a : op->fields) {
        args.push_back(gen(a));
    }
    CGExpr lhs = VarDecl::make(Type::make(op->child->getType()),
                               op->child->getName());
    code = VarAssign::make(lhs, Call::make(method_call, args));

    declared_adt.insert(op->child);
    used_adt.insert(op->parent);
}

void CodeGenerator::visit(const ComputeNode *op) {
    std::string func_call = op->f->getName();
    std::vector<CGExpr> args;
    for (auto a : op->f->getArguments()) {
        args.push_back(gen(a, op->new_ds));
    }

    code = VoidCall::make(Call::make(func_call, args));

    // Add the header.
    std::vector<std::string> func_header = op->f->getHeader();
    headers.insert(func_header.begin(), func_header.end());
}

void CodeGenerator::visit(const IntervalNode *op) {

    // First, lower the body of the interval node.
    std::vector<CGStmt> body;
    // In the case that the interval is mapped to a grid
    // variable, set it up.
    body.push_back(setGrid(op));
    // Continue lowering the body now.
    for (const auto &node : op->body) {
        this->visit(node);
        body.push_back(code);
    }
    code = Block::make(body);

    // If the variable is not mapped to the grid, we need to actually
    // wrap it in a for loop.
    if (!op->isMappedToGrid()) {
        Variable v = op->getIntervalVariable();
        CGStmt start = gen(op->start);
        CGExpr cond = gen(v < op->end);
        CGStmt step = gen(v += op->step);
        code = For::make(start, cond, step, code);
    }
}

void CodeGenerator::visit(const DefNode *op) {
    code = gen(op->assign);
}

void CodeGenerator::visit(const BlankNode *) {
    code = CGStmt();
}

#define VISIT_AND_DECLARE(op)          \
    void visit(const op##Node *node) { \
        this->visit(node->a);          \
        auto a = cg_e;                 \
        this->visit(node->b);          \
        cg_e = op::make(a, cg_e);      \
    }

CGExpr CodeGenerator::gen(Expr e) {

    struct ConvertToCode : public ExprVisitorStrict {
        ConvertToCode(CodeGenerator *cg)
            : cg(cg) {
        }
        using ExprVisitorStrict::visit;
        void visit(const LiteralNode *op) {
            cg_e = Literal::make(op->val, op->getDatatype());
        }
        void visit(const VariableNode *op) {
            cg_e = Var::make(op->name);
            cg->insertInUsed(op);
        }

        VISIT_AND_DECLARE(Add);
        VISIT_AND_DECLARE(Sub);
        VISIT_AND_DECLARE(Mul);
        VISIT_AND_DECLARE(Div);
        VISIT_AND_DECLARE(Mod);

        CodeGenerator *cg;
        CGExpr cg_e;
    };

    ConvertToCode cg(this);
    cg.visit(e);
    return cg.cg_e;
}

#define VISIT_AND_DECLARE_CONSTRAINT(op, name) \
    void visit(const op##Node *node) {         \
        auto a = cg->gen(node->a);             \
        auto b = cg->gen(node->b);             \
        cg_e = codegen::name::make(a, b);      \
    }

CGExpr CodeGenerator::gen(Constraint c) {

    struct ConvertToCode : public ConstraintVisitorStrict {
        ConvertToCode(CodeGenerator *cg)
            : cg(cg) {
        }
        using ConstraintVisitorStrict::visit;

        VISIT_AND_DECLARE_CONSTRAINT(Eq, Eq);
        VISIT_AND_DECLARE_CONSTRAINT(Neq, Neq);
        VISIT_AND_DECLARE_CONSTRAINT(Greater, Gt);
        VISIT_AND_DECLARE_CONSTRAINT(Less, Lt);
        VISIT_AND_DECLARE_CONSTRAINT(Geq, Gte);
        VISIT_AND_DECLARE_CONSTRAINT(Leq, Lte);
        VISIT_AND_DECLARE_CONSTRAINT(And, And);
        VISIT_AND_DECLARE_CONSTRAINT(Or, Or);

        CodeGenerator *cg;
        CGExpr cg_e;
    };

    ConvertToCode cg(this);
    cg.visit(c);
    return cg.cg_e;
}

CGStmt CodeGenerator::gen(Assign a) {
    if (!isa<Variable>(a.getA())) {
        throw error::UserError("Assignments can only be made to Variables");
    }

    Variable to_declare = to<Variable>(a.getA());
    // The variable has already been declared, we are just updating.
    if (declared.count(to_declare) > 0) {
        return VarAssign::make(
            gen(a.getA()),
            gen(a.getB()));
    }

    return VarAssign::make(
        declVar(to_declare),
        gen(a.getB()));
}

CGExpr CodeGenerator::gen(Argument a, const std::map<AbstractDataTypePtr, AbstractDataTypePtr> &replacements) {
    switch (a.getType()) {

    case DATA_STRUCTURE: {
        const DSArg *ds = to<DSArg>(a.get());
        // Track the fact that we are using this ADT
        used_adt.insert(ds->getADTPtr());
        // If the actual data-structure has been queried, then we need to make
        // sure that the queried name is used, not the original name.
        if (replacements.count(ds->getADTPtr()) > 0) {
            return Var::make(replacements.at(ds->getADTPtr())->getName());
        } else {
            return Var::make(ds->getADTPtr()->getName());
        }
        break;
    }

    case GERN_VARIABLE: {
        const VarArg *v = to<VarArg>(a.get());
        used.insert(v->getVar());
        return Var::make(v->getVar().getName());
    }

    default:
        throw error::InternalError("Unreachable");
        break;
    }
}

void CodeGenerator::insertInUsed(Variable v) {
    used.insert(v);
}

std::string CodeGenerator::getName() const {
    return name;
}

std::string CodeGenerator::getHookName() const {
    return hook_name;
}

std::vector<std::string> CodeGenerator::getArgumentOrder() const {
    return argument_order;
}

CGExpr CodeGenerator::declVar(Variable v) {
    declared.insert(v);
    return VarDecl::make(Type::make(v.getType().str()), v.getName());
}

static CGExpr genProp(const Grid::Property &p) {
    switch (p) {

    case Grid::Property::BLOCK_ID_X:
        return EscapeCGExpr::make("blockIdx.x");
    case Grid::Property::BLOCK_ID_Y:
        return EscapeCGExpr::make("blockIdx.y");
    case Grid::Property::BLOCK_ID_Z:
        return EscapeCGExpr::make("blockIdx.z");

    case Grid::Property::THREAD_ID_X:
        return EscapeCGExpr::make("threadIdx.x");
    case Grid::Property::THREAD_ID_Y:
        return EscapeCGExpr::make("threadIdx.y");
    case Grid::Property::THREAD_ID_Z:
        return EscapeCGExpr::make("threadIdx.z");

    case Grid::Property::BLOCK_DIM_X:
        return EscapeCGExpr::make("blockDim.x");
    case Grid::Property::BLOCK_DIM_Y:
        return EscapeCGExpr::make("blockDim.y");
    case Grid::Property::BLOCK_DIM_Z:
        return EscapeCGExpr::make("blockDim.z");

    default:
        throw error::InternalError("Undefined Grid Property Passed!");
    }
}

CGStmt CodeGenerator::setGrid(const IntervalNode *op) {
    // If not a grid mapping, do nothing.
    if (!op->isMappedToGrid()) {
        return BlankLine::make();
    }

    Variable interval_var = op->getIntervalVariable();
    Grid::Property property = interval_var.getBoundProperty();

    // This only works for ceiling.
    CGExpr divisor = gen(op->step);
    CGExpr dividend = gen(op->end) - gen(op->start.getB());
    auto ceil = (divisor + dividend - 1) / divisor;

    // Store the grid dimension that correspond with this mapping.
    if (property == Grid::Property::BLOCK_ID_X) {
        grid_dim.x = ceil;
    } else if (property == Grid::Property::BLOCK_ID_Y) {
        grid_dim.y = ceil;
    } else if (property == Grid::Property::BLOCK_ID_Z) {
        grid_dim.z = ceil;
    } else if (property == Grid::Property::THREAD_ID_X) {
        block_dim.x = ceil;
    } else if (property == Grid::Property::THREAD_ID_Y) {
        block_dim.y = ceil;
    } else if (property == Grid::Property::THREAD_ID_Z) {
        block_dim.z = ceil;
    } else {
        throw error::InternalError("Unreachable");
    }

    // Actually declare the variable to use the grid.
    return VarAssign::make(
        declVar(interval_var),
        // Add any shift factor specified in the interval.
        (genProp(interval_var.getBoundProperty()) * gen(op->step)) + gen(op->start.getB()));
}

}  // namespace codegen

}  // namespace gern