#ifndef ANNOT_ABSTRACT_NODES
#define ANNOT_ABSTRACT_NODES

#include "annotations/datatypes.h"
#include "utils/uncopyable.h"

namespace gern {

class ExprVisitorStrict;
class StmtVisitorStrict;

class ExprNode : private gern::Uncopyable {
public:
  ExprNode() = default;
  ExprNode(Datatype type) : datatype(type) {};
  virtual ~ExprNode() = default;

  virtual void accept(ExprVisitorStrict *) const = 0;
  Datatype getDatatype() const { return datatype; }

private:
  Datatype datatype;
};

class StmtNode : private gern::Uncopyable {
public:
  StmtNode() = default;
  virtual ~StmtNode() = default;
  virtual void accept(StmtVisitorStrict *) const = 0;
};

} // namespace gern

#endif