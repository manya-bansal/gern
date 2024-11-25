#ifndef GERN_VISITOR_H
#define GERN_VISITOR_H

#include "annotations/data_dependency_language.h"

namespace gern {

struct LiteralNode;
struct AddNode;
struct MulNode;
struct SubNode;
struct DivNode;
struct ModNode;

class ExprVisitorStrict {
public:
  virtual ~ExprVisitorStrict() = default;

  virtual void visit(Expr);
  virtual void visit(const LiteralNode *) = 0;
  virtual void visit(const AddNode *) = 0;
  virtual void visit(const SubNode *) = 0;
  virtual void visit(const MulNode *) = 0;
  virtual void visit(const DivNode *) = 0;
  virtual void visit(const ModNode *) = 0;
};

class Printer : public ExprVisitorStrict {
public:
  using ExprVisitorStrict::visit;
  Printer(std::ostream &os) : os(os) {}
  void visit(const LiteralNode *);
  virtual void visit(const AddNode *);
  virtual void visit(const SubNode *);
  virtual void visit(const MulNode *);
  virtual void visit(const DivNode *);
  virtual void visit(const ModNode *);

  std::ostream &os;
};
} // namespace gern
#endif