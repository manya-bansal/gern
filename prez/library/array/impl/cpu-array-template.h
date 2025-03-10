#pragma once

#include <cstring>
#include <iostream>
#include <stdlib.h>

namespace library {
namespace impl {

/**
 * @brief An ArrayCPU class to test out Gern's
 * codegen facilities.
 *
 */
template<int64_t Size>
class ArrayCPUTemplate {
public:
    template<int64_t Len>
    ArrayCPUTemplate<Len> query(int start) {
        ArrayCPUTemplate<Len> queried;
        for (int i = 0; i < Len; i++) {
            queried.data[i] = data[start + i];
        }
        return queried;
    }

    template<int64_t ToInsert, int64_t Len>
    void insert(int start, ArrayCPUTemplate<Len> to_insert) {
        for (int i = 0; i < ToInsert; i++) {
            data[start + i] = to_insert.data[i];
        }
    }

    void destroy() {
    }

    void vvals(float f) {
        for (int i = 0; i < Size; i++) {
            data[i] = f;
        }
    }

    void ascending() {
        for (int i = 0; i < Size; i++) {
            data[i] = i;
        }
    }

    float data[Size] = {0};
    int64_t size = Size;
};

template<int Len>
ArrayCPUTemplate<Len> temp_allocate() {
    return ArrayCPUTemplate<Len>();
}

template<int Size>
inline void add(ArrayCPUTemplate<Size> &a, ArrayCPUTemplate<Size> &b) {
    for (int64_t i = 0; i < Size; i++) {
        b.data[i] = a.data[i] + 1;
    }
}

}  // namespace impl
}  // namespace library