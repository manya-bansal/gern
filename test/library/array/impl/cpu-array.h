#pragma once

#include <cstring>
#include <iostream>
#include <stdlib.h>

namespace gern {
namespace impl {

/**
 * @brief An ArrayCPU class to test out Gern's
 * codegen facilities.
 *
 */

class ArrayCPU {
public:
    ArrayCPU(float *data, int64_t size)
        : data(data), size(size) {
    }
    ArrayCPU(int64_t size)
        : ArrayCPU((float *)calloc(size, sizeof(float)), size) {
    }
    static ArrayCPU allocate(int64_t start, int64_t len) {
        (void)start;
        return ArrayCPU(len);
    }
    ArrayCPU query(int64_t start, int64_t len) {
        return ArrayCPU(data + start, len);
    }
    void insert(int64_t start, int64_t len, ArrayCPU to_insert) {
        std::memcpy(data + start, to_insert.data, len);
    }

    void destroy() {
        free(data);
    }

    void vvals(float f) {
        for (int64_t i = 0; i < size; i++) {
            data[i] = f;
        }
    }

    void ascending() {
        for (int64_t i = 0; i < size; i++) {
            data[i] = (float)i;
        }
    }

    float *data;
    int64_t size;
};

[[maybe_unused]] static std::ostream &operator<<(std::ostream &os, const ArrayCPU &m) {
    os << "[";
    for (int64_t j = 0; j < m.size; j++) {
        os << m.data[j] << " ";
    }
    os << "]";
    return os;
}

inline void add(ArrayCPU a, ArrayCPU b) {
    for (int64_t i = 0; i < a.size; i++) {
        b.data[i] += a.data[i];
    }
}

inline void reduction(ArrayCPU a, ArrayCPU b) {
    for (int64_t i = 0; i < b.size; i++) {
        b.data[i] = 0;
        for (int64_t j = 0; j < a.size; j++) {
            b.data[i] += a.data[j];
        }
    }
}

inline void add_1(ArrayCPU a, ArrayCPU b) {
    for (int64_t i = 0; i < a.size; i++) {
        b.data[i] = a.data[i] + 1;
    }
}

template<int64_t Len>
inline void add1Template(ArrayCPU a, ArrayCPU b) {
    for (int64_t i = 0; i < Len; i++) {
        b.data[i] = a.data[i] + 1;
    }
}

}  // namespace impl
}  // namespace gern