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

CGStmt CodeGenerator::generate_code(Composable c) {
    // Lower each IR node one by one.
    ComposableLower lower(c);
    // Generate code the after lowering.
    bool is_device_call = false;
    top_level_codegen(lower.lower(), is_device_call);

    // Generate the hook.
    std::vector<CGStmt> hook_body;
    // Emit the const expr definitions.
    for (const auto &const_v : compute_func.template_args) {
        hook_body.push_back(gen(const_v = Expr(const_v.getInt64Val()), true));
    }

    DeclProperties properties{
        .is_const = false,
        .num_ref = 1,
        .num_ptr = 0,
    };
    // Get all the arguments out of a void**.
    for (size_t i = 0; i < compute_func.args.size(); i++) {
        Parameter param = compute_func.args[i];
        hook_body.push_back(
            VarAssign::make(declParameter(param, false, properties),
                            Deref::make(
                                Cast::make(Type::make(param.getType() + "*"),
                                           Var::make("args[" + std::to_string(i) + "]")))));
    }
    // Now, call the compute function.
    CGStmt hook_call = gen(compute_func.constructCall());
    if (is_device_call) {
        // Declare the grid and block dimensions.
        // hook_body.push_back(EscapeCGStmt::make("dim3 __grid_dim__(" + grid_dim.str() + ");"));
        // hook_body.push_back(EscapeCGStmt::make("dim3 __block_dim__(" + block_dim.str() + ");"));
        // hook_call = KernelLaunch::make(name, hook_args, call_template_vars, Var::make("__grid_dim__"), Var::make("__block_dim__"));
        // full_code.push_back(EscapeCGStmt::make("#include <cuda_runtime.h>"));
        throw error::InternalError("Unimplemented");
    }
    hook_body.push_back(hook_call);

    // Finally ready to generate the full file.
    std::vector<CGStmt> full_code;
    // Now, add the headers.
    for (const auto &h : headers) {
        full_code.push_back(EscapeCGStmt::make("#include \"" + h + "\""));
    }

    full_code.push_back(BlankLine::make());
    full_code.push_back(BlankLine::make());
    // Now, add the children (compute functions that must be declared).
    for (const auto &child : children) {
        full_code.push_back(child);
    }
    full_code.push_back(code);
    CGStmt hook = DeclFunc::make(hook_name, Type::make("void"),
                                 {VarDecl::make(Type::make("void**"), "args")},
                                 Block::make(hook_body));
    full_code.push_back(EscapeCGStmt::make("extern \"C\""));
    full_code.push_back(Scope::make(hook));

    return Block::make(full_code);
}

CGStmt CodeGenerator::top_level_codegen(LowerIR ir, bool is_device_launch) {
    this->visit(ir);  // code should contain the lowered IR.

    // Once we have visited the pipeline, we need to
    // wrap the body in a FunctionSignature interface.
    std::vector<Parameter> parameters;
    std::vector<Variable> template_arguments;
    // Add all the data-structures that haven't been declared
    // or allocated to the FunctionSignature arguments. These are the
    // true inputs, or the true outputs.
    std::vector<AbstractDataTypePtr> to_declare_adts;
    std::set_difference(used_adt.begin(), used_adt.end(),
                        declared_adt.begin(), declared_adt.end(),
                        std::back_inserter(to_declare_adts));

    for (const auto &ds : to_declare_adts) {
        argument_order.push_back(ds.getName());
        parameters.push_back(Parameter(ds));
    }

    // Declare all the variables that have been used, but have not been defined.
    for (const auto &v : used) {
        if (declared.contains(v)) {
            continue;
        }
        if (v.isConstExpr()) {
            if (!v.isBoundToInt64()) {
                throw error::UserError(v.getName() + " must be bound to an int64_t, it is a template parameter");
            }
            template_arguments.push_back(v);
            continue;
        }
        argument_order.push_back(v.getName());
        parameters.push_back(Parameter(v));
    }

    // The return type is always void, the output
    // is modified by reference.
    compute_func.name = name;
    compute_func.args = parameters;
    compute_func.template_args = template_arguments;
    compute_func.device = is_device_launch;

    // This generate the function declaration with the body.
    code = gen(compute_func, code);
    return code;
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
    CodeGenerator cg;
    bool is_device = false;
    // Push this back, so that we can declare it.
    children.push_back(cg.top_level_codegen(op->nodes, is_device));
    // Call the generated compute function.
    code = gen(cg.getComputeFunctionSignature().constructCall());
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

CGExpr CodeGenerator::declParameter(Parameter a,
                                    bool track,
                                    DeclProperties properties) {
    struct GenArgument : public ArgumentVisitorStrict {
        GenArgument(CodeGenerator *cg, bool track,

                    DeclProperties properties)
            : cg(cg), track(track),
              properties(properties) {
        }

        using ArgumentVisitorStrict::visit;
        void visit(const DSArg *ds) {
            gen_expr = cg->declADT(ds->getADTPtr(), track, properties);
        }
        void visit(const VarArg *v) {
            gen_expr = cg->declVar(v->getVar(), properties.is_const, track);
        }

        void visit(const ExprArg *v) {
            throw error::InternalError("unreachable");
        }

        CodeGenerator *cg;
        bool track;
        DeclProperties properties;
        CGExpr gen_expr;
    };

    GenArgument generate(this, track, properties);
    generate.visit(a);
    return generate.gen_expr;
}

CGExpr CodeGenerator::gen(Argument a) {

    struct GenArgument : public ArgumentVisitorStrict {
        GenArgument(CodeGenerator *cg)
            : cg(cg) {
        }
        using ArgumentVisitorStrict::visit;
        void visit(const DSArg *ds) {
            cg->used_adt.insert(ds->getADTPtr());
            gen_expr = cg->gen(ds->getADTPtr());
        }

        void visit(const VarArg *v) {
            gen_expr = cg->gen(v->getVar());
        }

        void visit(const ExprArg *v) {
            gen_expr = cg->gen(v->getExpr());
        }

        CodeGenerator *cg;
        CGExpr gen_expr;
    };

    GenArgument generate(this);
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
        return VarAssign::make(declParameter(f.output, true), call);
    }
    return VoidCall::make(call);
}

CGStmt CodeGenerator::gen(FunctionSignature f, CGStmt body) {
    std::vector<CGExpr> template_args_cg;
    std::vector<CGExpr> args_cg;
    std::vector<Parameter> args = f.args;
    std::vector<Variable> template_args = f.template_args;

    DeclProperties properties{
        .is_const = false,
        .num_ref = !f.device,
    };

    std::transform(
        args.begin(),
        args.end(),
        std::back_inserter(args_cg),
        [this, properties](const Parameter &a) {
            return declParameter(a, false, properties);
        });

    std::transform(
        template_args.begin(),
        template_args.end(),
        std::back_inserter(template_args_cg),
        [this, properties](const Parameter &a) {
            return declParameter(a, false, properties);
        });

    CGExpr return_type = Type::make(f.output.getType());
    return DeclFunc::make(f.name, return_type, args_cg, body, f.device, template_args_cg);
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

FunctionSignature CodeGenerator::getComputeFunctionSignature() const {
    return compute_func;
}

CGExpr CodeGenerator::declVar(Variable v, bool const_expr, bool track) {
    if (track) {
        declared.insert(v);
    }
    return VarDecl::make(Type::make((const_expr ? "constexpr " : "") +
                                    v.getType().str()),
                         v.getName());
}

CGExpr CodeGenerator::declWithAuto(AbstractDataTypePtr ds, bool track) {
    CGExpr decl = VarDecl::make(Type::make("auto"),
                                ds.getName());

    if (track) {
        declared_adt.insert(ds);
    }
    return decl;
}

CGExpr CodeGenerator::declADT(AbstractDataTypePtr ds,
                              bool track,
                              DeclProperties properties) {
    CGExpr decl = VarDecl::make(Type::make(ds.getType()),
                                ds.getName(), properties);

    if (track) {
        declared_adt.insert(ds);
    }
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