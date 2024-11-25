#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"

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

std::ostream &operator<<(std::ostream &os, const Expr &e) {
  Printer p{os};
  p.visit(e);
  return os;
}

} // namespace gern