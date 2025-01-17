#include "codegen/codegen.h"
#include "annotations/argument_visitor.h"
#include "annotations/data_dependency_language.h"
#include "annotations/grid.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/lower.h"
#include "utils/debug.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

CGStmt CodeGenerator::generate_code(Compose c) {
    // Lower each IR node one by one.
    ComposeLower lower(c);
    // Generate code the after lowering.
    return top_level_codegen(lower.lower(), c.isDeviceCall());
}

CGStmt CodeGenerator::top_level_codegen(LowerIR ir, bool is_device_launch) {
    this->visit(ir);  // code should contain the lowered IR.
    // Once we have visited the pipeline, we need to
    // wrap the body in a FunctionSignature interface.

    // Add all the data-structures that haven't been declared
    // or allocated to the FunctionSignature arguments. These are the
    // true inputs, or the true outputs.
    std::vector<AbstractDataTypePtr> to_declare_adts;
    std::set_difference(used_adt.begin(), used_adt.end(),
                        declared_adt.begin(), declared_adt.end(),
                        std::back_inserter(to_declare_adts));

    std::vector<Variable> to_declare_vars;
    std::vector<CGExpr> template_arg_vars;
    std::vector<CGExpr> call_template_vars;
    std::vector<CGStmt> hook_body;
    std::vector<Argument> arguments;
    std::vector<Expr> template_arguments;
    // Declare all the variables that have been used, but have not been defined.
    for (const auto &v : used) {
        if (declared.contains(v)) {
            continue;
        }
        if (const_expr_vars.contains(v)) {
            template_arg_vars.push_back(VarDecl::make(Type::make(v.getType().str()), v.getName()));
            call_template_vars.push_back(Var::make(v.getName()));
            template_arguments.push_back(v);
            if (!v.isBoundToInt64()) {
                throw error::UserError(v.getName() + " must be bound to an int64_t, it is a template parameter");
            }
            hook_body.push_back(gen(v = Expr(v.getInt64Val()), true));
            continue;
        }
        to_declare_vars.push_back(v);
    }

    std::vector<CGExpr> args;
    std::vector<CGExpr> hook_args;

    int num_args = 0;
    for (const auto &ds : to_declare_adts) {
        args.push_back(VarDecl::make(Type::make(ds.getType()), ds.getName(), false, !is_device_launch));
        hook_body.push_back(
            VarAssign::make(VarDecl::make(Type::make(ds.getType()), ds.getName(), false, !is_device_launch),
                            Deref::make(
                                Cast::make(Type::make(ds.getType() + "*"),
                                           Var::make("args[" + std::to_string(num_args) + "]")))));
        hook_args.push_back(Var::make(ds.getName()));
        argument_order.push_back(ds.getName());
        num_args++;
        arguments.push_back(ds);
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
        arguments.push_back(v);
    }

    // The return type is always void, the output
    // is modified by reference.
    code = DeclFunc::make(name, Type::make("void"), args, code, is_device_launch, template_arg_vars);

    compute_func.name = name;
    compute_func.args = arguments;
    compute_func.template_args = template_arguments;

    // Also make the FunctionSignature that is used as the entry point for
    // user code.
    std::vector<CGStmt> full_code;
    CGStmt hook_call = VoidCall::make(Call::make(name, hook_args, call_template_vars));
    if (is_device_launch) {
        // Declare the grid and block dimensions.
        hook_body.push_back(EscapeCGStmt::make("dim3 __grid_dim__(" + grid_dim.str() + ");"));
        hook_body.push_back(EscapeCGStmt::make("dim3 __block_dim__(" + block_dim.str() + ");"));
        hook_call = KernelLaunch::make(name, hook_args, call_template_vars, Var::make("__grid_dim__"), Var::make("__block_dim__"));
        full_code.push_back(EscapeCGStmt::make("#include <cuda_runtime.h>"));
    }

    hook_body.push_back(hook_call);

    CGStmt hook = DeclFunc::make(hook_name, Type::make("void"),
                                 {VarDecl::make(Type::make("void**"), "args")},
                                 Block::make(hook_body));

    // Now, add the children.
    for (const auto &child : children) {
        full_code.push_back(child);
    }

    // Now, add the headers.
    for (const auto &h : headers) {
        full_code.push_back(EscapeCGStmt::make("#include \"" + h + "\""));
    }

    full_code.push_back(BlankLine::make());
    full_code.push_back(BlankLine::make());
    full_code.push_back(code);
    full_code.push_back(EscapeCGStmt::make("extern \"C\""));
    full_code.push_back(Scope::make(hook));

    return Block::make(full_code);
}

void CodeGenerator::visit(const AllocateNode *op) {
    code = gen(op->f);
}

void CodeGenerator::visit(const FreeNode *op) {

    if (!declared_adt.contains(op->data)) {
        throw error::InternalError("Freeing a data-structure that hasn't been allocated??");
    }

    std::string method_call = op->data.getName() + ".destroy";
    code = VoidCall::make(Call::make(method_call, {}));
}

void CodeGenerator::visit(const InsertNode *op) {
    code = gen(op->f);
    used_adt.insert(op->parent);
}

void CodeGenerator::visit(const QueryNode *op) {
    code = gen(op->f);
    used_adt.insert(op->parent);
}

void CodeGenerator::visit(const ComputeNode *op) {
    code = gen(op->f);
    // Add the header.
    std::vector<std::string> func_header = op->headers;
    headers.insert(func_header.begin(), func_header.end());
}

void CodeGenerator::visit(const IntervalNode *op) {

    // First, lower the body of the interval node.
    std::vector<CGStmt> body;
    // In the case that the interval is mapped to a grid
    // variable, set it up.
    body.push_back(setGrid(op));
    // Continue lowering the body now.
    this->visit(op->body);
    body.push_back(code);
    code = Block::make(body);

    // If the variable is not mapped to the grid, we need to actually
    // wrap it in a for loop.
    if (!op->isMappedToGrid()) {
        Variable v = op->getIntervalVariable();
        CGStmt start = gen(op->start, false);  // Definitely not a constexpr.
        CGExpr cond = gen(v < op->end);
        CGStmt step = gen(v += op->step, false);  // Definitely not a constexpr.
        code = For::make(start, cond, step, code);
    }
}

void CodeGenerator::visit(const DefNode *op) {
    code = gen(op->assign, op->const_expr);
}

void CodeGenerator::visit(const BlankNode *) {
    code = CGStmt();
}

void CodeGenerator::visit(const BlockNode *op) {
    std::vector<CGStmt> block;
    for (const auto &node : op->ir_nodes) {
        this->visit(node);
        block.push_back(code);
    }
    code = Block::make(block);
}

void CodeGenerator::visit(const FunctionBoundary *op) {
    this->visit(op->nodes);
    // CodeGenerator cg;
    // bool is_device = false;
    // Push this back, so that we can declare it.
    // children.push_back(top_level_codegen(op->nodes, is_device));
    // code = gen(cg.getComputeFunction());
}

#define VISIT_AND_DECLARE(op)          \
    void visit(const op##Node *node) { \
        this->visit(node->a);          \
        auto a = cg_e;                 \
        this->visit(node->b);          \
        cg_e = op::make(a, cg_e);      \
    }

CGExpr CodeGenerator::gen(Expr e, bool const_expr) {

    struct ConvertToCode : public ExprVisitorStrict {
        ConvertToCode(CodeGenerator *cg, bool const_expr)
            : cg(cg), const_expr(const_expr) {
        }
        using ExprVisitorStrict::visit;
        void visit(const LiteralNode *op) {
            cg_e = Literal::make(op->val, op->getDatatype());
        }
        void visit(const ADTMemberNode *op) {
            cg_e = MetaData::make(cg->gen(op->ds), op->member);
        }
        void visit(const VariableNode *op) {
            cg_e = Var::make(op->name);
            cg->insertInUsed(op);
            if (const_expr) {
                cg->insertInConstExpr(op);
            }
        }

        VISIT_AND_DECLARE(Add);
        VISIT_AND_DECLARE(Sub);
        VISIT_AND_DECLARE(Mul);
        VISIT_AND_DECLARE(Div);
        VISIT_AND_DECLARE(Mod);

        CodeGenerator *cg;
        bool const_expr;
        CGExpr cg_e;
    };

    ConvertToCode cg(this, const_expr);
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

CGExpr CodeGenerator::gen(AbstractDataTypePtr ds) {
    used_adt.insert(ds);
    return Var::make(ds.getName());
}

CGStmt CodeGenerator::gen(Assign a, bool const_expr) {
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
        declVar(to_declare, const_expr),
        gen(a.getB()));
}

CGExpr CodeGenerator::gen(Argument a, bool lhs) {

    struct GenArgument : public ArgumentVisitorStrict {
        GenArgument(CodeGenerator *cg, bool lhs)
            : cg(cg), lhs(lhs) {
        }
        using ArgumentVisitorStrict::visit;
        void visit(const DSArg *ds) {
            cg->used_adt.insert(ds->getADTPtr());
            if (lhs) {
                gen_expr = cg->declADT(ds->getADTPtr());
            } else {
                gen_expr = cg->gen(ds->getADTPtr());
            }
        }
        void visit(const VarArg *v) {
            if (lhs) {
                gen_expr = cg->declVar(v->getVar(), false);
            } else {
                gen_expr = cg->gen(v->getVar());
            }
        }

        void visit(const ExprArg *v) {
            gen_expr = cg->gen(v->getExpr());
        }

        CodeGenerator *cg;
        bool lhs;
        CGExpr gen_expr;
    };
    GenArgument generate(this, lhs);
    generate.visit(a);
    return generate.gen_expr;
}

CGStmt CodeGenerator::gen(FunctionCall f) {
    std::vector<CGExpr> args;
    for (const auto &a : f.args) {
        args.push_back(gen(a));
    }
    std::vector<CGExpr> template_args;
    for (auto a : f.template_args) {
        template_args.push_back(gen(a, true));  // All of these are const_expr;
    }

    CGExpr call = Call::make(f.name, args, template_args);
    if (f.output.defined()) {
        return VarAssign::make(gen(f.output, true), call);
    }
    return VoidCall::make(call);
}

void CodeGenerator::insertInUsed(Variable v) {
    used.insert(v);
}

void CodeGenerator::insertInConstExpr(Variable v) {
    const_expr_vars.insert(v);
}

std::string CodeGenerator::getName() const {
    return name;
}

std::string CodeGenerator::getHeaders() const {
    return name;
}

std::string CodeGenerator::getHookName() const {
    return hook_name;
}

std::vector<std::string> CodeGenerator::getArgumentOrder() const {
    return argument_order;
}

FunctionCall CodeGenerator::getComputeFunction() const {
    return compute_func;
}

CGExpr CodeGenerator::declVar(Variable v, bool const_expr) {
    declared.insert(v);
    return VarDecl::make(Type::make((const_expr ? "constexpr " : "") +
                                    v.getType().str()),
                         v.getName());
}

CGExpr CodeGenerator::declADT(AbstractDataTypePtr ds) {
    CGExpr decl = VarDecl::make(Type::make("auto"),
                                ds.getName());
    declared_adt.insert(ds);
    return decl;
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
        declVar(interval_var, false),
        // Add any shift factor specified in the interval.
        (genProp(interval_var.getBoundProperty()) * gen(op->step)) + gen(op->start.getB()));
}

}  // namespace codegen

}  // namespace gern