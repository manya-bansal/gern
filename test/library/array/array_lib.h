#pragma once

#include <cstring>
#include <stdlib.h>

/**
 * @brief An array class to test out Gern's
 * codegen facilities.
 *
 */

class TestArray {
public:
    TestArray(float *data, int size)
        : data(data), size(size) {
    }
    TestArray(int size)
        : TestArray((float *)malloc(sizeof(float) * size), size) {
    }
    static TestArray allocate(int start, int len) {
        (void)start;
        return TestArray(len);
    }
    TestArray query(int start, int len) {
        return TestArray(data + start, len);
    }
    void insert(int start, int len, TestArray to_insert) {
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

void add(TestArray a, TestArray b) {
    for (int i = 0; i < a.size; i++) {
        b.data[i] += a.data[i];
    }
}