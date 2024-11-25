#ifndef GERN_DATA_DEP_LANG_H
#define GERN_DATA_DEP_LANG_H

#include "annotations/abstract_nodes.h"

namespace gern {

class Expr {
public:
  Expr(std::shared_ptr<const ExprNode> e) : node(e) {}

  Expr(uint8_t);
  Expr(uint16_t);
  Expr(uint32_t);
  Expr(uint64_t);
  Expr(int8_t);
  Expr(int16_t);
  Expr(int32_t);
  Expr(int64_t);
  Expr(float);
  Expr(double);
  Expr(const std::string &);

  std::shared_ptr<const ExprNode> getNode() { return node; }

private:
  std::shared_ptr<const ExprNode> node;
};

std::ostream &operator<<(std::ostream &os, const Expr &);
Expr operator+(const Expr &, const Expr &);
Expr operator-(const Expr &, const Expr &);
Expr operator*(const Expr &, const Expr &);
Expr operator/(const Expr &, const Expr &);
Expr operator%(const Expr &, const Expr &);

} // namespace gern
#endif