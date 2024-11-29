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
    : ConsumeMany(std::make_shared<const SubsetsNode>(inputs)) {}

Produces::Produces(Subset s) : Stmt(std::make_shared<const ProducesNode>(s)) {}

Consumes::Consumes(std::shared_ptr<const ConsumesNode> c) : Stmt(c) {}

ConsumeMany For(Variable v, Expr start, Expr end, Expr step, ConsumeMany body,
                bool parallel) {
  return ConsumeMany(std::make_shared<const ConsumesForNode>(
      v, start, end, step, body, parallel));
}

Allocates::Allocates(Expr reg, Expr smem)
    : Stmt(std::make_shared<const AllocatesNode>(reg, smem)) {}

Computes::Computes(Produces p, Consumes c, Allocates a)
    : Pattern(std::make_shared<const ComputesNode>(p, c, a)) {}

Pattern::Pattern(std::shared_ptr<const PatternNode> p) : Stmt(p) {}

Pattern For(Variable v, Expr start, Expr end, Expr step, Pattern body,
            bool parallel) {
  return Pattern(std::make_shared<const ComputesForNode>(v, start, end, step,
                                                         body, parallel));
}

} // namespace gern