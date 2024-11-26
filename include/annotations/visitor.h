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
struct VariableNode;
struct EqNode;
struct NeqNode;
struct LeqNode;
struct GeqNode;
struct LessNode;
struct GreaterNode;
struct AndNode;
struct OrNode;
struct ConstraintNode;


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
  virtual void visit(const VariableNode *) = 0;
  virtual void visit(const EqNode *) = 0;
  virtual void visit(const NeqNode *) = 0;
  virtual void visit(const LeqNode *) = 0;
  virtual void visit(const GeqNode *) = 0;
  virtual void visit(const LessNode *) = 0;
  virtual void visit(const GreaterNode *) = 0;
  virtual void visit(const OrNode *) = 0;
  virtual void visit(const AndNode *) = 0;
  virtual void visit(const ConstraintNode *) = 0;
};

class StmtVisitorStrict {
public:
  virtual ~StmtVisitorStrict() = default;

  virtual void visit(Stmt);
};

class Printer : public ExprVisitorStrict, public StmtVisitorStrict {
public:
  using ExprVisitorStrict::visit;
  using StmtVisitorStrict::visit;

  Printer(std::ostream &os) : os(os) {}
  void visit(const LiteralNode *);
  virtual void visit(const AddNode *);
  virtual void visit(const SubNode *);
  virtual void visit(const MulNode *);
  virtual void visit(const DivNode *);
  virtual void visit(const ModNode *);
  virtual void visit(const VariableNode *);
  virtual void visit(const EqNode *);
  virtual void visit(const NeqNode *);
  virtual void visit(const LeqNode *);
  virtual void visit(const GeqNode *);
  virtual void visit(const LessNode *);
  virtual void visit(const GreaterNode *);
  virtual void visit(const OrNode *);
  virtual void visit(const AndNode *);

  virtual void visit(const ConstraintNode *);

  std::ostream &os;
};

} // namespace gern
#endif