#ifndef ANNOT_ABSTRACT_NODES
#define ANNOT_ABSTRACT_NODES

#include "annotations/datatypes.h"
#include "utils/uncopyable.h"

namespace gern {

class ExprVisitorStrict;

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
} // namespace gern

#endif