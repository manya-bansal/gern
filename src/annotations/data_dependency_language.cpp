#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <set>

namespace gern {

Expr::Expr(uint8_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint16_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint32_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(uint64_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int8_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int16_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int32_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(int64_t val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(float val) : Expr(std::make_shared<const LiteralNode>(val)) {}
Expr::Expr(double val) : Expr(std::make_shared<const LiteralNode>(val)) {}

Variable::Variable(const std::string &name)
    : Expr(std::make_shared<const VariableNode>(name)) {}

std::ostream &operator<<(std::ostream &os, const Expr &e) {
  Printer p{os};
  p.visit(e);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Constraint &c) {
  Printer p{os};
  p.visit(c);
  return os;
}

#define DEFINE_BINARY_OPERATOR(CLASS_NAME, OPERATOR, NODE)                     \
  NODE operator OPERATOR(const Expr &a, const Expr &b) {                       \
    return CLASS_NAME(a, b);                                                   \
  }

DEFINE_BINARY_OPERATOR(Add, +, Expr)
DEFINE_BINARY_OPERATOR(Sub, -, Expr)
DEFINE_BINARY_OPERATOR(Mul, *, Expr)
DEFINE_BINARY_OPERATOR(Div, /, Expr)
DEFINE_BINARY_OPERATOR(Mod, %, Expr)
DEFINE_BINARY_OPERATOR(Eq, ==, Constraint)
DEFINE_BINARY_OPERATOR(Neq, !=, Constraint)
DEFINE_BINARY_OPERATOR(Leq, <=, Constraint)
DEFINE_BINARY_OPERATOR(Geq, >=, Constraint)
DEFINE_BINARY_OPERATOR(Less, <, Constraint)
DEFINE_BINARY_OPERATOR(Greater, >, Constraint)
DEFINE_BINARY_OPERATOR(And, &&, Constraint)
DEFINE_BINARY_OPERATOR(Or, ||, Constraint)

std::ostream &operator<<(std::ostream &os, const Stmt &s) {
  Printer p{os};
  p.visit(s);
  return os;
}

template <typename T> std::set<const VariableNode *> getVariables(T annot) {
  std::set<const VariableNode *> vars;
  match(annot,
        std::function<void(const VariableNode *, Matcher *)>(
            [&](const VariableNode *op, Matcher *ctx) { vars.insert(op); }));
  return vars;
}

Stmt Stmt::where(Constraint constraint) {
  auto stmtVars = getVariables(*this);
  auto constraintVars = getVariables(constraint);
  if (!std::includes(stmtVars.begin(), stmtVars.end(), constraintVars.begin(),
                     constraintVars.end())) {
    throw error::UserError("Putting constraints on variables that are not "
                           "present in statement's scope");
  }
  return Stmt(node, constraint);
}

#define DEFINE_BINARY_CONSTRUCTOR(CLASS_NAME, NODE)                            \
  CLASS_NAME::CLASS_NAME(Expr a, Expr b)                                       \
      : NODE(std::make_shared<const CLASS_NAME##Node>(a, b)) {}

DEFINE_BINARY_CONSTRUCTOR(Add, Expr)
DEFINE_BINARY_CONSTRUCTOR(Sub, Expr)
DEFINE_BINARY_CONSTRUCTOR(Div, Expr)
DEFINE_BINARY_CONSTRUCTOR(Mod, Expr)
DEFINE_BINARY_CONSTRUCTOR(Mul, Expr)

DEFINE_BINARY_CONSTRUCTOR(And, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Or, Constraint);

DEFINE_BINARY_CONSTRUCTOR(Eq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Neq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Leq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Geq, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Less, Constraint);
DEFINE_BINARY_CONSTRUCTOR(Greater, Constraint);

Subset::Subset(std::shared_ptr<const AbstractDataType> data,
               std::vector<Expr> mdFields)
    : Stmt(std::make_shared<const SubsetNode>(data, mdFields)) {}

Subsets::Subsets(const std::vector<Subset> &inputs)
    : Stmt(std::make_shared<const SubsetsNode>(inputs)) {}

For::For(Variable v, Expr start, Expr end, Expr step, Stmt body, bool parallel)
    : Stmt(std::make_shared<const ForNode>(v, start, end, step, body,
                                           parallel)) {}

Produces::Produces(Subset s) : Stmt(std::make_shared<const ProducesNode>(s)) {}

Consumes::Consumes(Stmt stmt)
    : Stmt(std::make_shared<const ConsumesNode>(stmt)) {}

Allocates::Allocates(Expr reg, Expr smem)
    : Stmt(std::make_shared<const AllocatesNode>(reg, smem)) {}

Computes::Computes(Produces p, Consumes c, Allocates a)
    : Stmt(std::make_shared<const ComputesNode>(p, c, a)) {}

bool isValidDataDependencyPattern(Stmt s) {

  bool found_compute_node = false;
  bool found_produces_node = false;
  bool found_consumes_node = false;
  bool ill_formed = false;

  std::set<std::shared_ptr<const ExprNode>> seen;

  match(s,
        std::function<void(const ComputesNode *, Matcher *)>(
            [&](const ComputesNode *op, Matcher *ctx) {
              if (found_compute_node) {
                DEBUG("Found two computes nodes");
                ill_formed = true;
              }
              found_compute_node = true;
              ctx->match(op->p);
              ctx->match(op->c);
              ctx->match(op->a);
            }),
        std::function<void(const ProducesNode *, Matcher *)>(
            [&](const ProducesNode *op, Matcher *ctx) {
              if (!found_compute_node || found_consumes_node) {
                DEBUG("Found produces node before computes node, or after a "
                      "consumes node");
                ill_formed = true;
              }
              if (found_produces_node) {
                DEBUG("Found two consumes nodes");
                ill_formed = true;
              }
              found_produces_node = true;
            }),
        std::function<void(const ConsumesNode *, Matcher *)>(
            [&](const ConsumesNode *op, Matcher *ctx) {
              if (!found_compute_node) {
                DEBUG("Found consumes node before computes node");
                ill_formed = true;
              }
              if (found_consumes_node) {
                DEBUG("Found two produces nodes");
                ill_formed = true;
              }
              found_consumes_node = true;
              ctx->match(op->stmt);
            }),
        std::function<void(const ForNode *, Matcher *)>(
            [&](const ForNode *op, Matcher *ctx) {
              if (seen.count(op->v.getNode()) != 0) {
                DEBUG("Same interval variable being used twice");
                ill_formed = true;
              }
              seen.insert(op->v.getNode());
              ctx->match(op->body);
            }));

  if (!found_compute_node || !found_consumes_node || !found_produces_node) {
    DEBUG("Did not find a consumes, produces or computes node");
    ill_formed = true;
  }

  return !ill_formed;
}

} // namespace gern