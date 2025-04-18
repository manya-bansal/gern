#include "annotations/visitor.h"
#include "annotations/expr_nodes.h"
#include "annotations/lang_nodes.h"
#include "annotations/rewriter.h"
#include "utils/printer.h"

namespace gern {

void ExprVisitorStrict::visit(Expr e) {
    e.accept(this);
}

void ConstraintVisitorStrict::visit(Constraint c) {
    c.accept(this);
}

void StmtVisitorStrict::visit(Stmt s) {
    s.accept(this);
}

void Printer::visit(Consumes p) {
    if (!p.defined()) {
        return;
    }
    util::printIdent(os, ident);
    os << " when consumes {"
       << "\n";
    ident++;
    p.accept(this);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(ConsumeMany many) {
    many.accept(this);
}

void Printer::visit(const LiteralNode *op) {
    os << (*op);
}
void Printer::visit(const VariableNode *op) {
    os << op->name;
    if (isLegalUnit(op->p)) {
        os << op->p;
    }
}

void Printer::visit(const ADTMemberNode *op) {
    os << op->ds
       << "." << op->member;
}

void Printer::visit(const GridDimNode *op) {
    os << op->dim;
}

#define DEFINE_PRINTER_METHOD(CLASS_NAME, OPERATOR)      \
    void Printer::visit(const CLASS_NAME *op) {          \
        os << "(" << op->a << #OPERATOR << op->b << ")"; \
    }

DEFINE_PRINTER_METHOD(AddNode, +)
DEFINE_PRINTER_METHOD(SubNode, -)
DEFINE_PRINTER_METHOD(MulNode, *)
DEFINE_PRINTER_METHOD(DivNode, /)
DEFINE_PRINTER_METHOD(ModNode, %)
DEFINE_PRINTER_METHOD(EqNode, ==)
DEFINE_PRINTER_METHOD(NeqNode, !=)
DEFINE_PRINTER_METHOD(LeqNode, <=)
DEFINE_PRINTER_METHOD(GeqNode, >=)
DEFINE_PRINTER_METHOD(LessNode, <)
DEFINE_PRINTER_METHOD(GreaterNode, >)
DEFINE_PRINTER_METHOD(AndNode, &&)
DEFINE_PRINTER_METHOD(OrNode, ||)
DEFINE_PRINTER_METHOD(AssignNode, =)

void Printer::visit(const SubsetNode *op) {
    util::printIdent(os, ident);
    os << op->data << " {"
       << "\n";
    ident++;
    int size_mf = op->mdFields.size();
    for (int i = 0; i < size_mf; i++) {
        util::printIdent(os, ident);
        os << op->mdFields[i];
        os << ((i != size_mf - 1) ? "," : "") << "\n";
    }
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const SubsetObjManyNode *op) {
    Printer p{os, ident};
    int size_mf = op->subsets.size();
    for (int i = 0; i < size_mf; i++) {
        p.visit(op->subsets[i]);
        os << ((i != size_mf - 1) ? ",\n" : "");
    }
}

void Printer::visit(const ProducesNode *op) {
    util::printIdent(os, ident);
    os << "produces {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->output);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ConsumesNode *op) {
    (void)op;
}

void Printer::visit(Allocates a) {
    if (!a.defined()) {
        util::printIdent(os, ident);
        os << "allocates()";
        return;
    }
    this->visit(a.ptr);
}

void Printer::visit(const AllocatesNode *op) {
    util::printIdent(os, ident);
    os << "allocates { register : " << op->reg << " , smem : " << op->smem << "}";
}

void Printer::visit(const ConsumesForNode *op) {
    util::printIdent(os, ident);
    os << "for ( " << op->start << " ; " << op->parameter << " ; "
       << op->step << " ) {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->body);
    ident--;
    os << "\n";
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ComputesForNode *op) {
    util::printIdent(os, ident);
    os << "for [ " << op->start << " : " << op->parameter << " : "
       << op->step << " ] {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->body);
    ident--;
    os << "\n";
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const ComputesNode *op) {
    util::printIdent(os, ident);
    os << "computes {"
       << "\n";
    ident++;
    Printer p{os, ident};
    p.visit(op->p);
    os << "\n";
    p.visit(op->c);
    os << "\n";
    p.visit(op->a);
    os << "\n";
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

void Printer::visit(const PatternNode *op) {
    (void)op;
}

void Printer::visit(const AnnotationNode *op) {
    this->visit(op->p);
    os << " @ ";
    util::iterable_printer(os, op->occupied, 0);
    util::printIdent(os, ident);
    os << " with constraints {";
    ident++;
    util::iterable_printer(os, op->constraints, ident, "\n");
    ident--;
    util::printIdent(os, ident);
    os << "}";
}

#define DEFINE_BINARY_VISITOR_METHOD(CLASS_NAME)     \
    void AnnotVisitor::visit(const CLASS_NAME *op) { \
        this->visit(op->a);                          \
        this->visit(op->b);                          \
    }

void AnnotVisitor::visit(const LiteralNode *) {
}

void AnnotVisitor::visit(const GridDimNode *) {
}

DEFINE_BINARY_VISITOR_METHOD(AddNode)
DEFINE_BINARY_VISITOR_METHOD(SubNode)
DEFINE_BINARY_VISITOR_METHOD(DivNode)
DEFINE_BINARY_VISITOR_METHOD(MulNode)
DEFINE_BINARY_VISITOR_METHOD(ModNode)

DEFINE_BINARY_VISITOR_METHOD(AndNode)
DEFINE_BINARY_VISITOR_METHOD(OrNode)

DEFINE_BINARY_VISITOR_METHOD(EqNode)
DEFINE_BINARY_VISITOR_METHOD(NeqNode)
DEFINE_BINARY_VISITOR_METHOD(LeqNode)
DEFINE_BINARY_VISITOR_METHOD(GeqNode)
DEFINE_BINARY_VISITOR_METHOD(LessNode)
DEFINE_BINARY_VISITOR_METHOD(GreaterNode)

DEFINE_BINARY_VISITOR_METHOD(AssignNode)

void AnnotVisitor::visit(const VariableNode *) {
}

void AnnotVisitor::visit(const ADTMemberNode *) {
}

void AnnotVisitor::visit(const SubsetNode *op) {
    for (const auto &field : op->mdFields) {
        this->visit(field);
    }
}

void AnnotVisitor::visit(const SubsetObjManyNode *op) {
    for (const auto &subset : op->subsets) {
        this->visit(subset);
    }
}

void AnnotVisitor::visit(const ProducesNode *op) {
    this->visit(op->output);
}

void AnnotVisitor::visit(const ConsumesNode *op) {
    (void)op;
}

void AnnotVisitor::visit(const PatternNode *op) {
    (void)op;
}

void AnnotVisitor::visit(const AnnotationNode *op) {
    this->visit(op->p);
}

void AnnotVisitor::visit(const AllocatesNode *op) {
    this->visit(op->reg);
    this->visit(op->smem);
}

void AnnotVisitor::visit(const ConsumesForNode *op) {
    this->visit(op->start);
    this->visit(op->parameter);
    this->visit(op->step);
    this->visit(op->body);
}

void AnnotVisitor::visit(const ComputesForNode *op) {
    this->visit(op->start);
    this->visit(op->parameter);
    this->visit(op->step);
    this->visit(op->body);
}

void AnnotVisitor::visit(const ComputesNode *op) {
    this->visit(op->p);
    this->visit(op->c);
    this->visit(op->a);
}

// class ExprRewriterStrict
Expr Rewriter::rewrite(Expr e) {
    if (e.defined()) {
        e.accept(this);
    } else {
        expr = Expr();
    }
    return expr;
}

Stmt Rewriter::rewrite(Stmt s) {
    if (s.defined()) {
        s.accept(this);
    } else {
        stmt = Stmt();
    }
    return stmt;
}

Constraint Rewriter::rewrite(Constraint c) {
    if (c.defined()) {
        c.accept(this);
    } else {
        where = Constraint();
    }
    return where;
}

void Rewriter::visit(const VariableNode *op) {
    expr = op;
}

void Rewriter::visit(const ADTMemberNode *op) {
    expr = ADTMember(op->ds, op->member, op->const_expr);
}

void Rewriter::visit(const LiteralNode *op) {
    expr = op;
}

void Rewriter::visit(const GridDimNode *op) {
    expr = GridDim(op->dim);
}

void Rewriter::visit(const ConsumesNode *op) {
    (void)op;
}

void Rewriter::visit(const PatternNode *op) {
    (void)op;
}

void Rewriter::visit(const SubsetNode *op) {
    std::vector<Expr> rw_expr;
    for (size_t i = 0; i < op->mdFields.size(); i++) {
        rw_expr.push_back(this->rewrite(op->mdFields[i]));
    }
    stmt = SubsetObj(op->data, rw_expr);
}

void Rewriter::visit(const SubsetObjManyNode *op) {
    std::vector<SubsetObj> rw_subsets;
    for (size_t i = 0; i < op->subsets.size(); i++) {
        rw_subsets.push_back(to<SubsetObj>(this->rewrite(op->subsets[i])));
    }
    stmt = SubsetObjMany(rw_subsets);
}

void Rewriter::visit(const ProducesNode *op) {
    SubsetObj rw_subset = to<SubsetObj>(this->rewrite(op->output));
    std::vector<Variable> vars;
    std::vector<Expr> expr_vars = rw_subset.getFields();
    for (const auto &e : expr_vars) {
        vars.push_back(to<Variable>(e));
    }
    stmt = Produces::Subset(rw_subset.getDS(), vars);
}

void Rewriter::visit(const AllocatesNode *op) {
    Expr rw_reg = op->reg;
    Expr rw_smem = op->smem;
    stmt = Allocates(rw_reg, rw_smem);
}

void Rewriter::visit(const ConsumesForNode *op) {
    Assign rw_start = to<Assign>(this->rewrite(op->start));
    Expr rw_parameter = this->rewrite(op->parameter);
    Variable rw_step = to<Variable>(this->rewrite(op->step));
    ConsumeMany rw_body = to<ConsumeMany>(this->rewrite(op->body));
    stmt = Consumes(new const ConsumesForNode(rw_start,
                                              rw_parameter, rw_step,
                                              rw_body, op->parallel));
}

void Rewriter::visit(const ComputesForNode *op) {
    Assign rw_start = to<Assign>(this->rewrite(op->start));
    Expr rw_parameter = this->rewrite(op->parameter);
    Variable rw_step = to<Variable>(this->rewrite(op->step));
    Pattern rw_body = to<Pattern>(this->rewrite(op->body));
    stmt = Pattern(new const ComputesForNode(rw_start,
                                             rw_parameter, rw_step,
                                             rw_body, op->parallel));
}

void Rewriter::visit(const ComputesNode *op) {
    Produces rw_produces = to<Produces>(this->rewrite(op->p));
    Consumes rw_consumes = to<Consumes>(this->rewrite(op->c));
    if (op->a.defined()) {
        Allocates rw_allocates = to<Allocates>(this->rewrite(op->a));
        stmt = Computes(rw_produces, rw_consumes, rw_allocates);
    } else {
        stmt = Computes(rw_produces, rw_consumes);
    }
}

void Rewriter::visit(const AnnotationNode *op) {
    Pattern rw_pattern = to<Pattern>(this->rewrite(op->p));
    std::vector<Constraint> op_constraints = op->constraints;
    std::vector<Constraint> rw_constraints;
    for (const auto &c : op_constraints) {
        rw_constraints.push_back(this->rewrite(c));
    }
    stmt = Annotation(rw_pattern, op->occupied, rw_constraints);
}

#define DEFINE_BINARY_REWRITER_METHOD(CLASS_NAME, PARENT, VAR) \
    void Rewriter::visit(const CLASS_NAME *op) {               \
        Expr rw_a = this->rewrite(op->a);                      \
        Expr rw_b = this->rewrite(op->b);                      \
        VAR = PARENT(new const CLASS_NAME(rw_a, rw_b));        \
    }

DEFINE_BINARY_REWRITER_METHOD(AddNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(SubNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(DivNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(MulNode, Expr, expr)
DEFINE_BINARY_REWRITER_METHOD(ModNode, Expr, expr)

DEFINE_BINARY_REWRITER_METHOD(AndNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(OrNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(EqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(NeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(LeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(GeqNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(LessNode, Constraint, where)
DEFINE_BINARY_REWRITER_METHOD(GreaterNode, Constraint, where)

DEFINE_BINARY_REWRITER_METHOD(AssignNode, Stmt, stmt)

Consumes mimicConsumes(Pattern p, std::vector<SubsetObj> input_subsets) {
    ConsumeMany consumes = SubsetObjMany(input_subsets);
    match(p, std::function<void(const ConsumesForNode *, Matcher *)>(
                 [&](const ConsumesForNode *op, Matcher *ctx) {
                     ctx->match(op->body);
                     consumes = Reducible(op->start, op->parameter, op->step, consumes);
                 }));
    return consumes;
}

Pattern mimicComputes(Pattern p, Computes computes) {
    Pattern pattern = computes;
    match(p, std::function<void(const ComputesForNode *, Matcher *)>(
                 [&](const ComputesForNode *op, Matcher *ctx) {
                     ctx->match(op->body);
                     pattern = Tileable(op->start, op->parameter, op->step, pattern);
                 }));
    return pattern;
}

}  // namespace gern