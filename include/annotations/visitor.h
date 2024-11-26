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

struct SubsetNode;
struct SubsetsNode;
struct ProducesNode;
struct ConsumesNode;
struct ForNode;
struct ComputesNode;

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
  virtual void visit(const SubsetNode *) = 0;
  virtual void visit(const SubsetsNode *) = 0;
  virtual void visit(const ProducesNode *) = 0;
  virtual void visit(const ConsumesNode *) = 0;
  virtual void visit(const ForNode *) = 0;
  virtual void visit(const ComputesNode *) = 0;
};

class Printer : public ExprVisitorStrict, public StmtVisitorStrict {
public:
  using ExprVisitorStrict::visit;
  using StmtVisitorStrict::visit;

  Printer(std::ostream &os, int ident = 0) : os(os), ident(ident) {}
  void visit(const LiteralNode *);
  void visit(const AddNode *);
  void visit(const SubNode *);
  void visit(const MulNode *);
  void visit(const DivNode *);
  void visit(const ModNode *);
  void visit(const VariableNode *);
  void visit(const EqNode *);
  void visit(const NeqNode *);
  void visit(const LeqNode *);
  void visit(const GeqNode *);
  void visit(const LessNode *);
  void visit(const GreaterNode *);
  void visit(const OrNode *);
  void visit(const AndNode *);
  void visit(const ConstraintNode *);

  void visit(const SubsetNode *);
  void visit(const SubsetsNode *);
  void visit(const ProducesNode *);
  void visit(const ConsumesNode *);
  void visit(const ForNode *);
  void visit(const ComputesNode *);

private:
  std::ostream &os;
  int ident;
};

} // namespace gern
#endif