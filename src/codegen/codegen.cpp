#include "annotations/data_dependency_language.h"

#include "codegen/codegen.h"

#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

CGStmt CodeGenerator::generate_code(const Pipeline &p) {
    this->visit(p);
    // Once we have visited the pipeline, we need to define
    // the function.

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
    for (const auto &ds : to_declare_adts) {
        args.push_back(VarDecl::make(Type::make(ds->getType()), ds->getName()));
    }
    for (const auto &v : to_declare_vars) {
        args.push_back(VarDecl::make(Type::make(v.getType().getString()), v.getName()));
    }
    // The return type is always void, we return
    // by modifying the function.
    code = DeclFunc::make(name, Type::make("void"), args, code);
    std::cout << code << std::endl;
    return code;
}

void CodeGenerator::visit(const Pipeline &p) {
    for (const auto &node : p.getIRNodes()) {
        this->visit(node);
    }
}

void CodeGenerator::visit(const AllocateNode *op) {
    std::string method_call = op->data->getType() + "::allocate";
    std::vector<CGExpr> args;
    for (const auto &a : op->fields) {
        args.push_back(genCodeExpr(a));
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

    std::string method_call = op->data->getName() + ".free";
    code = VoidCall::make(Call::make(method_call, {}));
}

void CodeGenerator::visit(const InsertNode *op) {
    std::string method_call = op->parent->getName() + ".insert";
    std::vector<CGExpr> args;
    for (const auto &a : op->fields) {
        args.push_back(genCodeExpr(a));
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
        args.push_back(genCodeExpr(a));
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
        switch (a.getType()) {

        case DATA_STRUCTURE: {
            const DSArg *ds = to<DSArg>(a.get());
            // Track the fact that we are using this ADT
            used_adt.insert(ds->getADTPtr());
            // If the actual data-structure has been queried, then we need to make
            // sure that the queried name is used, not the original name.
            if (op->new_ds.count(ds->getADTPtr()) > 0) {
                args.push_back(Var::make(op->new_ds.at(ds->getADTPtr())->getName()));
            } else {
                args.push_back(Var::make(ds->getADTPtr()->getName()));
            }
            break;
        }

        default:
            throw error::InternalError("Unreachable");
            break;
        }
    }

    code = VoidCall::make(Call::make(func_call, args));
}

void CodeGenerator::visit(const IntervalNode *op) {

    // First, lower the body of the interval node.
    std::vector<CGStmt> body;
    for (const auto &node : op->body) {
        this->visit(node);
        body.push_back(code);
    }

    // Finally wrap the lowered body in a for loop.
    CGStmt start = genCodeExpr(op->start);
    CGExpr cond = genCodeExpr(op->end);
    CGStmt step = genCodeExpr(op->step);
    code = For::make(start, cond, step, Block::make(body));
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

CGExpr CodeGenerator::genCodeExpr(Expr e) {

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
        auto a = cg->genCodeExpr(node->a);     \
        auto b = cg->genCodeExpr(node->b);     \
        cg_e = codegen::name::make(a, b);      \
    }

CGExpr CodeGenerator::genCodeExpr(Constraint c) {

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

CGStmt CodeGenerator::genCodeExpr(Assign a) {
    if (!isa<Variable>(a.getA())) {
        throw error::UserError("Assignments can only be made to Variables");
    }

    Variable to_declare = to<Variable>(a.getA());
    // The variable has already been declared, we are just updating.
    if (declared.count(to_declare) > 0) {
        return VarAssign::make(
            genCodeExpr(a.getA()),
            genCodeExpr(a.getB()));
    }

    declared.insert(to<Variable>(a.getA()));
    return VarAssign::make(
        VarDecl::make(Type::make(to_declare.getType().getString()), to_declare.getName()),
        genCodeExpr(a.getB()));
}

void CodeGenerator::insertInUsed(Variable v) {
    used.insert(v);
}

}  // namespace codegen

}  // namespace gern