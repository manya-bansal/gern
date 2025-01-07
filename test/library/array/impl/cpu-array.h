#pragma once

#include <cstring>
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
    ArrayCPU(float *data, int size)
        : data(data), size(size) {
    }
    ArrayCPU(int size)
        : ArrayCPU((float *)malloc(sizeof(float) * size), size) {
    }
    static ArrayCPU allocate(int start, int len) {
        (void)start;
        return ArrayCPU(len);
    }
    ArrayCPU query(int start, int len) {
        return ArrayCPU(data + start, len);
    }
    void insert(int start, int len, ArrayCPU to_insert) {
        std::memcpy(data + start, to_insert.data, len);
    }

    void destroy() {
        free(data);
    }

    void vvals(float f) {
        for (int i = 0; i < size; i++) {
            data[i] = f;
        }
    }

    float *data;
    int size;
};

inline void add(ArrayCPU a, ArrayCPU b) {
    for (int64_t i = 0; i < a.size; i++) {
        b.data[i] += a.data[i];
    }
}

}  // namespace impl
}  // namespace gern