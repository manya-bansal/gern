#pragma once

#include <complex>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

namespace gern {

/// A basic annotation type. These can be boolean, integer, unsigned integer,
/// float or complex float at different precisions.
class Datatype {
public:
    /// The kind of type this object represents.
    enum Kind {
        Bool,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
        Complex64,
        Complex128,
        Undefined  /// Undefined type
    };

    /// Construct an undefined type.
    Datatype();

    /// Construct a taco basic type with default bit widths.
    Datatype(Kind);

    /// Return the kind of type this object represents.
    Kind getKind() const;

    /// Functions that return true if the type is the given type.
    /// @{
    bool isUInt() const;
    bool isInt() const;
    bool isFloat() const;
    bool isComplex() const;
    bool isBool() const;
    std::string str() const;

    /// Returns the number of bytes required to store one element of this type.
    int getNumBytes() const;

    /// Returns the number of bits required to store one element of this type.
    int getNumBits() const;

private:
    Kind kind;
};

std::ostream &operator<<(std::ostream &, const Datatype &);
std::ostream &operator<<(std::ostream &, const Datatype::Kind &);
bool operator==(const Datatype &a, const Datatype &b);
bool operator!=(const Datatype &a, const Datatype &b);

extern const Datatype Bool;
Datatype UInt(int bits = sizeof(unsigned int) * 8);
extern const Datatype UInt8;
extern const Datatype UInt16;
extern const Datatype UInt32;
extern const Datatype UInt64;
Datatype Int(int bits = sizeof(int) * 8);
extern const Datatype Int8;
extern const Datatype Int16;
extern const Datatype Int32;
extern const Datatype Int64;
Datatype Float(int bits = sizeof(double) * 8);
extern const Datatype Float32;
extern const Datatype Float64;
Datatype Complex(int bits);
extern const Datatype Complex64;
extern const Datatype Complex128;

Datatype max_type(Datatype a, Datatype b);

template<typename T>
inline Datatype type() {
    std::cerr << "Unsupported type";
    return Int32;
}

template<>
inline Datatype type<bool>() {
    return Bool;
}

template<>
inline Datatype type<unsigned char>() {
    return UInt(sizeof(char) * 8);
}

template<>
inline Datatype type<unsigned short>() {
    return UInt(sizeof(short) * 8);
}

template<>
inline Datatype type<unsigned int>() {
    return UInt(sizeof(int) * 8);
}

template<>
inline Datatype type<unsigned long>() {
    return UInt(sizeof(long) * 8);
}

template<>
inline Datatype type<unsigned long long>() {
    return UInt(sizeof(long long) * 8);
}

template<>
inline Datatype type<char>() {
    return Int(sizeof(char) * 8);
}

template<>
inline Datatype type<short>() {
    return Int(sizeof(short) * 8);
}

template<>
inline Datatype type<int>() {
    return Int(sizeof(int) * 8);
}

template<>
inline Datatype type<long>() {
    return Int(sizeof(long) * 8);
}

template<>
inline Datatype type<long long>() {
    return Int(sizeof(long long) * 8);
}

template<>
inline Datatype type<int8_t>() {
    return Int8;
}

template<>
inline Datatype type<float>() {
    return Float32;
}

template<>
inline Datatype type<double>() {
    return Float64;
}

template<>
inline Datatype type<std::complex<float>>() {
    return Complex64;
}

template<>
inline Datatype type<std::complex<double>>() {
    return Complex128;
}

}  // namespace gern