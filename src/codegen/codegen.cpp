#include "codegen/codegen.h"
#include "annotations/argument_visitor.h"
#include "annotations/data_dependency_language.h"
#include "annotations/grid.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "codegen/concretize.h"
#include "codegen/finalizer.h"
#include "codegen/helpers/assert_device_properties.h"
#include "codegen/helpers/check_last_error.h"
#include "codegen/lower.h"
#include "utils/debug.h"
#include "utils/error.h"
#include <algorithm>
namespace gern {
namespace codegen {

CGStmt CodeGenerator::generate_code(Composable c) {
    // Lower each IR node one by one.
    Concretize concretizer(c);
    // Generate code the after lowering.
    bool is_device_call = c.isDeviceLaunch();
    auto ir = concretizer.concretize();
    Finalizer finalizer(ir);
    auto final_ir = finalizer.finalize();

    top_level_codegen(final_ir, is_device_call);

    // Generate the hook.
    std::vector<CGStmt> hook_body;
    // Emit the const expr definitions.
    for (const auto &const_v : compute_func.template_args) {
        hook_body.push_back(gen(const_v = Expr(const_v.getInt64Val())));
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

    // Generate calls to assertions before calling the function.
    if (is_device_call) {
        hook_body.push_back(gen(
            helpers::assert_device_properties(
                compute_func.grid.x,
                compute_func.grid.y,
                compute_func.grid.z,
                compute_func.block.x,
                compute_func.block.y,
                compute_func.block.z,
                getCurrentVal(Grid::Dim::WARP_DIM_X),
                getCurrentVal(Grid::Dim::WARP_DIM_Y),
                getCurrentVal(Grid::Dim::WARP_DIM_Z),
                // If smem_size is not defined,
                // then we will just run with default,
                // just check trivial condition against 0.
                compute_func.smem_size.defined() ? compute_func.smem_size : Expr(0))));
    }
    // Now, call the compute function.
    CGStmt hook_call = gen(compute_func.constructCall());
    hook_body.push_back(hook_call);

    if (is_device_call) {
        hook_body.push_back(gen(helpers::check_last_error()));
    }

    // Finally ready to generate the full file.
    std::vector<CGStmt> full_code;
    // Now, add the headers.
    for (const auto &h : headers) {
        full_code.push_back(EscapeCGStmt::make("#include \"" + h + "\""));
    }
    if (is_device_call) {
        full_code.push_back(EscapeCGStmt::make("#include <cuda_runtime.h>"));
        full_code.push_back(EscapeCGStmt::make(helpers::assert_device_constraints_decl));
        full_code.push_back(EscapeCGStmt::make(helpers::check_last_error_decl));
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

    dims_defined[Grid::Dim::BLOCK_DIM_X].insert(1);
    dims_defined[Grid::Dim::BLOCK_DIM_X].scope();

    dims_defined[Grid::Dim::BLOCK_DIM_Y].insert(1);
    dims_defined[Grid::Dim::BLOCK_DIM_Y].scope();

    dims_defined[Grid::Dim::BLOCK_DIM_Z].insert(1);
    dims_defined[Grid::Dim::BLOCK_DIM_Z].scope();

    dims_defined[Grid::Dim::GRID_DIM_X].insert(1);
    dims_defined[Grid::Dim::GRID_DIM_X].scope();

    dims_defined[Grid::Dim::GRID_DIM_Y].insert(1);
    dims_defined[Grid::Dim::GRID_DIM_Y].scope();

    dims_defined[Grid::Dim::GRID_DIM_Z].insert(1);
    dims_defined[Grid::Dim::GRID_DIM_Z].scope();

    dims_defined[Grid::Dim::WARP_DIM_X].insert(32);
    dims_defined[Grid::Dim::WARP_DIM_X].scope();
    dims_defined[Grid::Dim::WARP_DIM_Y].insert(32);
    dims_defined[Grid::Dim::WARP_DIM_Y].scope();
    dims_defined[Grid::Dim::WARP_DIM_Z].insert(32);
    dims_defined[Grid::Dim::WARP_DIM_Z].scope();

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
                        std::back_inserter(to_declare_adts),
                        std::less<AbstractDataTypePtr>());

    for (const auto &ds : to_declare_adts) {
        parameters.push_back(Parameter(ds));
    }

    LaunchArguments block_dim{
        .x = getCurrentVal(Grid::Dim::BLOCK_DIM_X),
        .y = getCurrentVal(Grid::Dim::BLOCK_DIM_Y),
        .z = getCurrentVal(Grid::Dim::BLOCK_DIM_Z),
    };

    LaunchArguments grid_dim{
        .x = getCurrentVal(Grid::Dim::GRID_DIM_X),
        .y = getCurrentVal(Grid::Dim::GRID_DIM_Y),
        .z = getCurrentVal(Grid::Dim::GRID_DIM_Z),
    };

    // Also (fake gen block and grid params, otherwise they will not be tracked as used).
    gen(block_dim.x);
    gen(block_dim.y);
    gen(block_dim.z);
    gen(grid_dim.x);
    gen(grid_dim.y);
    gen(grid_dim.z);
    gen(smem_size);

    // Declare all the variables that have been used, but have not been defined.
    for (const auto &v : used) {
        if (declared.contains(v)) {
            continue;
        }
        if (v.isConstExpr()) {
            if (!v.isBound()) {
                throw error::UserError(v.getName() +
                                       " must be bound to an int64_t, it is a template parameter");
            }
            template_arguments.push_back(v);
            continue;
        }
        parameters.push_back(Parameter(v));
    }

    // The return type is always void, the output
    // is modified by reference.
    compute_func.name = name;

    if (this->ordered_parameters.has_value() && !same_parameters(this->ordered_parameters.value(), parameters)) {
        throw error::UserError("provided ordered arguments dont match the parameters needed for the function");
    }
    compute_func.args = this->ordered_parameters.value_or(parameters);
    this->argument_order = get_parameter_names(compute_func.args);

    compute_func.template_args = template_arguments;
    compute_func.device = is_device_launch;
    compute_func.block = block_dim;
    compute_func.grid = grid_dim;
    compute_func.smem_size = smem_size;

    if (is_device_launch) {
        compute_func.access = GLOBAL;
        // push back helpers.
    } else {
        compute_func.access = HOST;
    }

    // This generate the function declaration with the body.
    // Make the grid assertions at the top level.
    CGStmt code_temp = code;
    std::vector<CGStmt> stmts;

    // for (const auto &dim : dims_defined) {
    //     stmts.push_back(assertGrid(dim.first));
    // }

    stmts.push_back(code_temp);
    code = gen(compute_func, Block::make(stmts));
    return code;
}

void CodeGenerator::visit(const AllocateNode *op) {
    code = gen(op->f);
}

void CodeGenerator::visit(const FreeNode *op) {

    if (!declared_adt.contains(op->call.data)) {
        throw error::InternalError("Freeing a data-structure that hasn't been allocated??");
    }

    code = gen(op->call);
}

void CodeGenerator::visit(const InsertNode *op) {
    code = gen(op->call);
    used_adt.insert(op->call.data);
}

void CodeGenerator::visit(const QueryNode *op) {
    code = gen(op->call);
    used_adt.insert(op->parent);
}

void CodeGenerator::visit(const ComputeNode *op) {
    code = gen(op->f);
    // Add the header.
    std::vector<std::string> func_header = op->headers;
    headers.insert(func_header.begin(), func_header.end());
}

static CGExpr genProp(const Grid::Unit &p) {
    switch (p) {

    case Grid::Unit::BLOCK_X:
        return EscapeCGExpr::make("blockIdx.x");
    case Grid::Unit::BLOCK_Y:
        return EscapeCGExpr::make("blockIdx.y");
    case Grid::Unit::BLOCK_Z:
        return EscapeCGExpr::make("blockIdx.z");

    case Grid::Unit::THREAD_X:
        return EscapeCGExpr::make("threadIdx.x");
    case Grid::Unit::THREAD_Y:
        return EscapeCGExpr::make("threadIdx.y");
    case Grid::Unit::THREAD_Z:
        return EscapeCGExpr::make("threadIdx.z");

    case Grid::Unit::WARP_X:
        return EscapeCGExpr::make("(threadIdx.x / 32)");
    case Grid::Unit::WARP_Y:
        return EscapeCGExpr::make("(threadIdx.y / 32)");
    case Grid::Unit::WARP_Z:
        return EscapeCGExpr::make("(threadIdx.z / 32)");

    case Grid::Unit::THREAD_X_IN_WRAPS:
        return EscapeCGExpr::make("(threadIdx.x % 32)");
    case Grid::Unit::THREAD_Y_IN_WRAPS:
        return EscapeCGExpr::make("(threadIdx.y % 32)");
    case Grid::Unit::THREAD_Z_IN_WRAPS:
        return EscapeCGExpr::make("(threadIdx.z % 32)");

    default:
        throw error::InternalError("Undefined Grid unit Passed!");
    }
}

void CodeGenerator::visit(const IntervalNode *op) {

    // First, lower the body of the interval node.
    std::vector<CGStmt> body;
    // In the case that the interval is mapped to a grid
    // variable, set it up.
    // Continue lowering the body now.
    Expr divisor = op->step;
    Expr dividend = op->end - op->start.getB();
    Expr ceil = (divisor + dividend - 1) / divisor;

    if (op->isMappedToGrid()) {
        dims_defined[getDim(op->p)].scope();
    }

    this->visit(op->body);
    CGStmt body_code = code;

    if (op->isMappedToGrid()) {

        Expr first = 1;
        auto dims = dims_defined[getDim(op->p)].pop();
        if (dims.size() > 0) {
            // set this to the puter dimension.
            first = *dims.begin();
        }

        if (getLevel(op->p) == Grid::Level::WARPS) {
            ceil = ceil * 32;
        }

        Variable interval_var = op->getIntervalVariable();
        dims_defined[getDim(op->p)].insert(first * ceil);

        body.push_back(VarAssign::make(
            declVar(interval_var, false),
            // Add any shift factor specified in the interval.
            (((genProp(op->p) / gen(first)) % gen(ceil)) * gen(op->step)) + gen(op->start.getB())));
    }

    body.push_back(body_code);
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

void CodeGenerator::visit(const AssertNode *op) {
    // Set up the current values for the grid dims.
    std::map<Grid::Dim, Expr> dims_defined_copy;
    for (auto &dim : dims_defined) {
        dims_defined_copy[dim.first] = getCurrentVal(dim.first);
    }

    Constraint rw_constraint = replaceDim(op->constraint, dims_defined_copy);
    std::string name = (isConstExpr(rw_constraint)) ?
                           "static_assert" :  // If both A and B are const exprs, generate a static assert.
                           "assert";          // Otherwise generate a normal assert.
    code = VoidCall::make(Call::make(name, {gen(rw_constraint)}));
}

void CodeGenerator::visit(const GridDeclNode *op) {
    gen(op->v);
    dims_defined[op->dim].insert(op->v);
    code = BlankLine::make();
}

void CodeGenerator::visit(const SharedMemoryDeclNode *op) {
    smem_size = op->size;
    code = BlankLine::make();
}

void CodeGenerator::visit(const OpaqueCall *op) {
    code = gen(op->f);
    headers.insert(op->headers.begin(), op->headers.end());
}

void CodeGenerator::visit(const BlankNode *) {
    code = BlankLine::make();
}

void CodeGenerator::visit(const BlockNode *op) {
    std::vector<CGStmt> block;
    for (const auto &node : op->ir_nodes) {
        this->visit(node);
        block.push_back(code);
    }
    code = Block::make(block);
}

CGExpr CodeGenerator::gen(const Grid::Dim &p) {
    return gen(getCurrentVal(p));
}

#define VISIT_AND_DECLARE(op)          \
    void visit(const op##Node *node) { \
        this->visit(node->a);          \
        auto a = cg_e;                 \
        this->visit(node->b);          \
        cg_e = op::make(a, cg_e);      \
    }

CGExpr CodeGenerator::gen(Expr e) {

    if (!e.defined()) {
        return CGExpr();
    }

    struct ConvertToCode : public ExprVisitorStrict {
        ConvertToCode(CodeGenerator *cg)
            : cg(cg) {
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
        }
        void visit(const GridDimNode *node) {
            cg_e = cg->gen(cg->getCurrentVal(node->dim));
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
        declVar(to_declare, isConstExpr(a.getB()) && const_expr),
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

        void visit(const ExprArg *e) {
            gen_expr = cg->declVar(e->getVar(), properties.is_const, track);
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

CGStmt CodeGenerator::gen(MethodCall call) {
    FunctionCall f = call.call;
    f.name = call.data.getName() + "." + f.name;
    return gen(f);
}

CGStmt CodeGenerator::gen(FunctionCall f) {
    std::vector<CGExpr> args;
    for (const auto &a : f.args) {
        args.push_back(gen(a));
    }
    std::vector<CGExpr> template_args;
    for (auto a : f.template_args) {
        template_args.push_back(gen(a));  // All of these are const_expr;
    }

    CGExpr call = Call::make(f.name, args, template_args);
    std::vector<CGStmt> stmt;
    // If it is a global function, then also set up the grid.
    if (f.access == GLOBAL) {
        std::string grid_name = getUniqueName("grid");
        std::string block_name = getUniqueName("block");

        LaunchArguments grid = f.grid.constructDefaults();
        LaunchArguments block = f.block.constructDefaults();
        CGExpr dim3_type = Type::make("dim3");

        // Define the grid dimensions.
        stmt.push_back(VarAssign::make(
            VarDecl::make(dim3_type, grid_name, DeclProperties()),
            Call::make("dim3", {gen(grid.x), gen(grid.y), gen(grid.z)})));
        // Define the block dimensions.
        stmt.push_back(VarAssign::make(
            VarDecl::make(dim3_type, block_name, DeclProperties()),
            Call::make("dim3", {gen(block.x), gen(block.y), gen(block.z)})));

        // Define the specialized function.
        std::string func_sp_name = getUniqueName("function_sp");
        stmt.push_back(VarAssign::make(
            VarDecl::make(Type::make("auto"), func_sp_name, DeclProperties()),
            SpecializedFunction::make(f.name, template_args)));

        // Define the size of shared memory if applicable.
        if (f.smem_size.defined()) {
            stmt.push_back(VoidCall::make(Call::make("cudaFuncSetAttribute",
                                                     {EscapeCGExpr::make(func_sp_name),
                                                      EscapeCGExpr::make("cudaFuncAttributeMaxDynamicSharedMemorySize"),
                                                      gen(f.smem_size)})));
        }
        // Has been specialized, use that.
        call = KernelLaunch::make(func_sp_name, args, {},
                                  Var::make(grid_name),
                                  Var::make(block_name),
                                  gen(f.smem_size));
    }

    if (f.output.defined()) {
        stmt.push_back(VarAssign::make(declParameter(f.output, true), call));
    } else {
        stmt.push_back(VoidCall::make(call));
    }

    return Block::make(stmt);
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

std::string CodeGenerator::getName() const {
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

void CodeGenerator::updateGrid(Grid::Dim dim, Expr expr) {
    dims_defined[dim].insert(expr);
}

Expr CodeGenerator::getCurrentVal(const Grid::Dim &dim) {
    if (dims_defined[dim].front().size() == 0) {
        if (getLevel(dim) == Grid::Level::THREADS_WARPS) {
            return 32;  // default warp size.
        }
        return 1;
    }
    return *dims_defined[dim].front().begin();
}

CGStmt CodeGenerator::assertGrid(const Grid::Dim &dim) {
    std::vector<CGStmt> stmts;
    const auto &exprs = dims_defined[dim].front();
    if (exprs.empty()) {
        return BlankLine::make();
    }

    Expr first = *exprs.begin();
    for (auto it = std::next(exprs.begin()); it != exprs.end(); ++it) {
        auto new_const = new const AssertNode(first == *it);
        visit(new_const);
        stmts.push_back(code);
    }

    return Block::make(stmts);
}

}  // namespace codegen

}  // namespace gern
