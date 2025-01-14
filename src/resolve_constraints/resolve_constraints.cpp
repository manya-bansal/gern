#include <ginac/ginac.h>

#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "resolve_constraints/resolve_constraints.h"
#include "utils/debug.h"
#include "utils/error.h"

namespace gern {
namespace resolve {

#define VISIT_AND_DECLARE(op, operation)   \
    void visit(const op *node) {           \
        this->visit(node->a);              \
        auto a = ginacExpr;                \
        this->visit(node->b);              \
        ginacExpr = a operation ginacExpr; \
    }

#define COMPLAIN(op)             \
    void visit(const op *node) { \
        (void)node;              \
        assert("bad");           \
    }

struct GinacLess {
    bool operator()(const GiNaC::symbol &a, const GiNaC::symbol &b) const {
        return GiNaC::ex_is_less()(a, b);
    }
};

typedef std::map<Expr, GiNaC::symbol> ExprToSymbolMap;
typedef std::map<GiNaC::symbol, Expr, GinacLess> SymbolToExprMap;

// Helper FunctionSignature to convert an equality constraint to a GiNaC
// expression. Currently, only equality constraints are accepted.
static GiNaC::ex convertToGinac(Eq q, ExprToSymbolMap names) {
    struct ExprToGinac : public ExprVisitorStrict {
        ExprToGinac(ExprToSymbolMap names)
            : names(names) {
        }
        using ExprVisitorStrict::visit;

        void visit(const VariableNode *op) {
            if (names.find(Variable(op)) == names.end()) {
                throw error::InternalError("Map does not contain a symbol for variable");
            }
            ginacExpr = names[op];
        }

        void visit(const ADTMemberNode *op) {
            if (names.find(ADTMember(op)) == names.end()) {
                throw error::InternalError("Map does not contain a symbol for ADT member");
            }
            ginacExpr = names[op];
        }

        void visit(const LiteralNode *op) {
            switch (op->getDatatype().getKind()) {
            case Datatype::Int64:
                ginacExpr = op->getVal<int64_t>();
                break;
            case Datatype::Int32:
                ginacExpr = op->getVal<int32_t>();
                break;
            default:
                throw error::InternalError("Unimplemented");
                break;
            }
        }

        VISIT_AND_DECLARE(AddNode, +);
        VISIT_AND_DECLARE(SubNode, -);
        VISIT_AND_DECLARE(MulNode, *);
        VISIT_AND_DECLARE(DivNode, /);
        COMPLAIN(ModNode);

        GiNaC::ex ginacExpr;
        ExprToSymbolMap names;
    };

    ExprToGinac convert_a{names};
    convert_a.visit(q.getA());
    GiNaC::ex a = convert_a.ginacExpr;
    ExprToGinac convert_b{names};
    convert_b.visit(q.getB());
    GiNaC::ex b = convert_b.ginacExpr;

    return a == b;
}

// Helper FunctionSignature to convert an GiNaC expression to a Gern
// expression.
static Expr convertToGern(GiNaC::ex ginacExpr, SymbolToExprMap variables) {
    struct GinacToExpr : public GiNaC::visitor,
                         public GiNaC::symbol::visitor,
                         public GiNaC::add::visitor,
                         public GiNaC::mul::visitor,
                         public GiNaC::numeric::visitor,
                         public GiNaC::power::visitor {
        GinacToExpr(SymbolToExprMap names)
            : names(names) {
        }

        void visit(const GiNaC::numeric &n) {
            if (n.is_integer()) {
                e = Expr(n.to_int());
            }
            if (n.is_real()) {
                e = Expr(n.to_double());
            } else {
                throw error::InternalError("Unimplemented");
            }
        }

        void visit(const GiNaC::symbol &s) {
            if (names.count(s) == 0) {
                throw error::InternalError("Map does not contain a symbol for GiNac::symbol");
            }
            e = names[s];
        }

        void visit(const GiNaC::add &a) {
            a.op(0).accept(*this);
            Expr test = e;
            for (size_t i = 1; i < a.nops(); i++) {
                a.op(i).accept(*this);
                test = test + e;
            }
            e = test;
        }

        void visit(const GiNaC::mul &a) {
            a.op(0).accept(*this);
            Expr test = e;
            for (size_t i = 1; i < a.nops(); i++) {
                a.op(i).accept(*this);
                test = test * e;
            }
            e = test;
        }

        void visit(const GiNaC::power &p) {
            p.op(0).accept(*this);

            // We should always have an integer degree.
            int degree = p.degree(p.op(0));
            Expr temp = e;

            for (int i = 1; i < degree; i++) {
                temp = temp * temp;
            }

            if (p.op(1).info(GiNaC::info_flags::negative)) {
                e = 1 / temp;
            } else {
                e = temp;
            }
        }

        SymbolToExprMap names;
        Expr e;
    };

    GinacToExpr convertor{variables};
    ginacExpr.accept(convertor);

    return convertor.e;
}

Expr solve(Eq eq, Variable v) {
    // Generate a GiNaC symbol for each variable node that we want to lower.
    ExprToSymbolMap symbols;
    match(eq, std::function<void(const VariableNode *)>([&](const VariableNode *op) { symbols[op] = GiNaC::symbol(op->name); }));

    // Convert the Gern equations into GiNaC equations.
    DEBUG(convertToGinac(eq, symbols));

    // Track all the symbols that we would like solutions for (may be an
    // overestimate).
    SymbolToExprMap variables;
    for (const auto &var : symbols) {
        variables[var.second] = var.first;
    }

    // Solve the equations.
    GiNaC::ex solution = GiNaC::lsolve(convertToGinac(eq, symbols), symbols[v]);

    // Convert back into a Gern expressions.
    return convertToGern(solution, variables);
}

}  // namespace resolve
}  // namespace gern