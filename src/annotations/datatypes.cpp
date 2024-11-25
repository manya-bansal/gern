#include "annotations/datatypes.h"

using namespace std;

namespace gern {

Datatype::Datatype() : kind(Undefined) {}

Datatype::Datatype(Kind kind) : kind(kind) {}

Datatype::Kind Datatype::getKind() const { return this->kind; }

bool Datatype::isBool() const { return getKind() == Bool; }

bool Datatype::isUInt() const {
  return getKind() == UInt8 || getKind() == UInt16 || getKind() == UInt32 ||
         getKind() == UInt64;
}

bool Datatype::isInt() const {
  return getKind() == Int8 || getKind() == Int16 || getKind() == Int32 ||
         getKind() == Int64;
}

bool Datatype::isFloat() const {
  return getKind() == Float32 || getKind() == Float64;
}

bool Datatype::isComplex() const {
  return getKind() == Complex64 || getKind() == Complex128;
}

std::string Datatype::getString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

Datatype max_type(Datatype a, Datatype b) {
  if (a == b) {
    return a;
  } else if (a.isComplex() || b.isComplex()) {
    if (a == Complex128 || b == Complex128 || a == Float64 || b == Float64) {
      return Complex128;
    } else {
      return Complex64;
    }
  } else if (a.isFloat() || b.isFloat()) {
    if (a == Float64 || b == Float64) {
      return Float64;
    } else {
      return Float32;
    }
  } else {
    if (a.isInt() || b.isInt()) {
      return Int((a.getNumBits() > b.getNumBits()) ? a.getNumBits()
                                                   : b.getNumBits());
    } else {
      return UInt((a.getNumBits() > b.getNumBits()) ? a.getNumBits()
                                                    : b.getNumBits());
    }
  }
}

int Datatype::getNumBytes() const { return (getNumBits() + 7) / 8; }

int Datatype::getNumBits() const {
  switch (getKind()) {
  case Bool:
    return sizeof(bool);
  case UInt8:
  case Int8:
    return 8;
  case UInt16:
  case Int16:
    return 16;
  case UInt32:
  case Int32:
  case Float32:
    return 32;
  case UInt64:
  case Int64:
  case Float64:
  case Complex64:
    return 64;
  case Complex128:
  default:
    std::cerr << "Bits for data type not set: " << getKind() << std::endl;
    return -1;
  }
}

std::ostream &operator<<(std::ostream &os, const Datatype &type) {
  if (type.isBool())
    os << "bool";
  else if (type.isInt())
    os << "int" << type.getNumBits() << "_t";
  else if (type.isUInt())
    os << "uint" << type.getNumBits() << "_t";
  else if (type == Datatype::Float32)
    os << "float";
  else if (type == Datatype::Float64)
    os << "double";
  else if (type == Datatype::Complex64)
    os << "float complex";
  else if (type == Datatype::Complex128)
    os << "double complex";
  else
    os << "Undefined";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Datatype::Kind &kind) {
  switch (kind) {
  case Datatype::Bool:
    os << "Bool";
    break;
  case Datatype::UInt8:
    os << "UInt8";
    break;
  case Datatype::UInt16:
    os << "UInt16";
    break;
  case Datatype::UInt32:
    os << "UInt32";
    break;
  case Datatype::UInt64:
    os << "UInt64";
    break;
  case Datatype::Int8:
    os << "Int8";
    break;
  case Datatype::Int16:
    os << "Int16";
    break;
  case Datatype::Int32:
    os << "Int32";
    break;
  case Datatype::Int64:
    os << "Int64";
    break;
  case Datatype::Float32:
    os << "Float32";
    break;
  case Datatype::Float64:
    os << "Float64";
    break;
  case Datatype::Complex64:
    os << "Complex64";
    break;
  case Datatype::Complex128:
    os << "Complex128";
    break;
  case Datatype::Undefined:
    os << "Undefined";
    break;
  }
  return os;
}

bool operator==(const Datatype &a, const Datatype &b) {
  return a.getKind() == b.getKind();
}

bool operator!=(const Datatype &a, const Datatype &b) {
  return a.getKind() != b.getKind();
}

Datatype Bool = Datatype(Datatype::Bool);

Datatype UInt(int bits) {
  switch (bits) {
  case 8:
    return Datatype(Datatype::UInt8);
  case 16:
    return Datatype(Datatype::UInt16);
  case 32:
    return Datatype(Datatype::UInt32);
  case 64:
    return Datatype(Datatype::UInt64);
  default:
    std::cerr << bits << " bits not supported for datatype UInt";
    return Datatype(Datatype::UInt32);
  }
}

Datatype UInt8 = Datatype(Datatype::UInt8);
Datatype UInt16 = Datatype(Datatype::UInt16);
Datatype UInt32 = Datatype(Datatype::UInt32);
Datatype UInt64 = Datatype(Datatype::UInt64);

Datatype Int(int bits) {
  switch (bits) {
  case 8:
    return Datatype(Datatype::Int8);
  case 16:
    return Datatype(Datatype::Int16);
  case 32:
    return Datatype(Datatype::Int32);
  case 64:
    return Datatype(Datatype::Int64);
  default:
    std::cerr << bits << " bits not supported for datatype Int";
    return Datatype(Datatype::Int32);
  }
}

Datatype Int8 = Datatype(Datatype::Int8);
Datatype Int16 = Datatype(Datatype::Int16);
Datatype Int32 = Datatype(Datatype::Int32);
Datatype Int64 = Datatype(Datatype::Int64);

Datatype Float(int bits) {
  switch (bits) {
  case 32:
    return Datatype(Datatype::Float32);
  case 64:
    return Datatype(Datatype::Float64);
  default:
    std::cerr << bits << " bits not supported for datatype Float";
    return Datatype(Datatype::Float64);
  }
}

Datatype Float32 = Datatype(Datatype::Float32);
Datatype Float64 = Datatype(Datatype::Float64);

Datatype Complex(int bits) {
  switch (bits) {
  case 64:
    return Datatype(Datatype::Complex64);
  case 128:
    return Datatype(Datatype::Complex128);
  default:
    std::cerr << bits << " bits not supported for datatype Complex";
    return Datatype(Datatype::Complex128);
  }
}

Datatype Complex64 = Datatype(Datatype::Complex64);
Datatype Complex128 = Datatype(Datatype::Complex128);

} // namespace gern