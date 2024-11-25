#ifndef GERN_LANG_NODES_H
#define GERN_LANG_NODES_H

#include "annotations/abstract_nodes.h"
#include "annotations/visitor.h"

#include <any>

namespace gern {

struct LiteralNode : public ExprNode {
  template <typename T>
  explicit LiteralNode(T val) : ExprNode(type<T>()), val(val) {}

  void accept(ExprVisitorStrict *v) const override { v->visit(this); }

  template <typename T> T getVal() const { return std::any_cast<T>(val); }

  std::any val;
};

std::ostream &operator<<(std::ostream &os, const LiteralNode &);

} // namespace gern

#endif