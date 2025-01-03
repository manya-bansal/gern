#include "codegen/codegen.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

CGStmt CodeGenerator::generate_code(const Pipeline &p) {
    this->visit(p);
    std::cout << p << std::endl;
    std::cout << code << std::endl;
    return CGStmt{};
}

void CodeGenerator::visit(const Pipeline &p) {
    for (const auto &node : p.getIRNodes()) {
        std::cout << "Here!" << std::endl;
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
}

void CodeGenerator::visit(const FreeNode *op) {
    std::string method_call = op->data->getName() + ".free";
    code = VoidCall::make(Call::make(method_call, {}));
}

void CodeGenerator::visit(const InsertNode *op) {
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
}

void CodeGenerator::visit(const ComputeNode *op) {
    std::string func_call = op->f->getName();
    std::vector<CGExpr> args;
    for (auto a : op->f->getArguments()) {
        switch (a.getType()) {

        case DATA_STRUCTURE: {
            const DSArg *ds = to<DSArg>(a.get());
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

    // Finally wrap the lowered body in an interval node.
    CGStmt start = genCodeExpr(op->start);
    CGExpr cond = genCodeExpr(op->end);
    CGStmt step = genCodeExpr(op->step);
    code = For::make(start, cond, step, Block::make(body));
}

void CodeGenerator::visit(const BlankNode *op) {
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
        using ExprVisitorStrict::visit;
        void visit(const LiteralNode *op) {
            cg_e = Literal::make(op->val, op->getDatatype());
        }
        void visit(const VariableNode *op) {
            cg_e = Var::make(op->name);
        }

        VISIT_AND_DECLARE(Add);
        VISIT_AND_DECLARE(Sub);
        VISIT_AND_DECLARE(Mul);
        VISIT_AND_DECLARE(Div);
        VISIT_AND_DECLARE(Mod);

        CGExpr cg_e;
    };
    ConvertToCode cg;
    cg.visit(e);
    return cg.cg_e;
}

#define VISIT_AND_DECLARE_CONSTRAINT(op, name)        \
    void visit(const op##Node *node) {                \
        auto a = CodeGenerator::genCodeExpr(node->a); \
        auto b = CodeGenerator::genCodeExpr(node->b); \
        cg_e = codegen::name::make(a, b);             \
    }

CGExpr CodeGenerator::genCodeExpr(Constraint c) {

    struct ConvertToCode : public ConstraintVisitorStrict {
        using ConstraintVisitorStrict::visit;

        VISIT_AND_DECLARE_CONSTRAINT(Eq, Eq);
        VISIT_AND_DECLARE_CONSTRAINT(Neq, Neq);
        VISIT_AND_DECLARE_CONSTRAINT(Greater, Gt);
        VISIT_AND_DECLARE_CONSTRAINT(Less, Lt);
        VISIT_AND_DECLARE_CONSTRAINT(Geq, Gte);
        VISIT_AND_DECLARE_CONSTRAINT(Leq, Lte);
        VISIT_AND_DECLARE_CONSTRAINT(And, And);
        VISIT_AND_DECLARE_CONSTRAINT(Or, Or);

        CGExpr cg_e;
    };
    ConvertToCode cg;
    cg.visit(c);
    return cg.cg_e;
}

CGStmt CodeGenerator::genCodeExpr(Assign a) {
    return VarAssign::make(genCodeExpr(a.getA()), genCodeExpr(a.getB()));
}

}  // namespace codegen

}  // namespace gern