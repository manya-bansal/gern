#include "annotations/lang_nodes.h"
#include "annotations/datatypes.h"

namespace gern {
std::ostream &operator<<(std::ostream &os, const LiteralNode &op) {
    switch (op.getDatatype().getKind()) {
    case Datatype::Kind::UInt64:
        os << op.getVal<uint64_t>();
        break;
    case Datatype::Kind::UInt32:
        os << op.getVal<uint32_t>();
        break;
    case Datatype::Kind::UInt16:
        os << op.getVal<uint16_t>();
        break;
    case Datatype::Kind::Int64:
        os << op.getVal<int64_t>();
        break;
    case Datatype::Kind::Int32:
        os << op.getVal<int32_t>();
        break;
    case Datatype::Kind::Int16:
        os << op.getVal<int16_t>();
        break;
    case Datatype::Kind::Int8:
        os << op.getVal<int8_t>();
        break;
    case Datatype::Kind::Float32:
        os << op.getVal<float>();
        break;
    case Datatype::Kind::Float64:
        os << op.getVal<double>();
        break;
    case Datatype::Kind::Complex64:
        os << op.getVal<std::complex<float>>();
        break;
    case Datatype::Kind::Complex128:
        os << op.getVal<std::complex<double>>();
        break;
    default:
        break;
    }

    return os;
}

std::vector<Variable> ProducesNode::getFieldsAsVars() const {
    std::vector<Variable> vars;
    std::vector<Expr> expr_vars = output.getFields();
    for (const auto &e : expr_vars) {
        vars.push_back(to<Variable>(e));
    }
    return vars;
}

}  // namespace gern