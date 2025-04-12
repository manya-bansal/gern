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

bool isSameValue(const LiteralNode &a, const LiteralNode &b) {
    if (a.getDatatype() != b.getDatatype()) {
        return false;
    }

    switch (a.getDatatype().getKind()) {
    case Datatype::Kind::UInt64:
        return a.getVal<uint64_t>() == b.getVal<uint64_t>();
    case Datatype::Kind::UInt32:
        return a.getVal<uint32_t>() == b.getVal<uint32_t>();
    case Datatype::Kind::UInt16:
        return a.getVal<uint16_t>() == b.getVal<uint16_t>();
    case Datatype::Kind::Int64:
        return a.getVal<int64_t>() == b.getVal<int64_t>();
    case Datatype::Kind::Int32:
        return a.getVal<int32_t>() == b.getVal<int32_t>();
    case Datatype::Kind::Int16:
        return a.getVal<int16_t>() == b.getVal<int16_t>();
    case Datatype::Kind::Int8:
        return a.getVal<int8_t>() == b.getVal<int8_t>();
    case Datatype::Kind::Float32:
        return a.getVal<float>() == b.getVal<float>();
    case Datatype::Kind::Float64:
        return a.getVal<double>() == b.getVal<double>();
    case Datatype::Kind::Complex64:
        return a.getVal<std::complex<float>>() == b.getVal<std::complex<float>>();
    case Datatype::Kind::Complex128:
        return a.getVal<std::complex<double>>() == b.getVal<std::complex<double>>();
    default:
        throw std::runtime_error("Unknown datatype");
        return false;
    }
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