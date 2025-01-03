#include "codegen/codegen.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"

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
}

void CodeGenerator::visit(const ComputeNode *op) {
}

void CodeGenerator::visit(const IntervalNode *op) {

    // First, lower the body of the interval node.
    std::vector<CGStmt> body;
    for (const auto &node : op->body) {
        this->visit(node);
        body.push_back(code);
    }

    CGStmt start = VarAssign::make(genCodeExpr(op->start), 0);
    // CGStmt cond =
    // Finally wrap the lowered body in an interval node.
    // code = For::make(genCodeExpr(op->start), genCodeExpr(op->cond), genCodeExpr(op->step))
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

}  // namespace codegen

}  // namespace gern