#ifndef GERN_VISITOR_H
#define GERN_VISITOR_H

#include "annotations/data_dependency_language.h"

namespace gern {

struct LiteralNode;

class ExprVisitorStrict {
public:
  virtual ~ExprVisitorStrict() = default;

  virtual void visit(Expr);
  virtual void visit(const LiteralNode *) = 0;
};

class Printer : public ExprVisitorStrict {
public:
  using ExprVisitorStrict::visit;
  Printer(std::ostream &os) : os(os) {}
  void visit(const LiteralNode *);

  std::ostream &os;
};
} // namespace gern
#endif