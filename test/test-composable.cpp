#include "codegen/lower.h"
#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(ComposableTest, Simple) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    auto call = Call(
        add_1.construct(inputDS, tempDS),
        add_1.construct(tempDS, outputDS));

    std::cout << call.getAnnotation() << std::endl;

    call = For(outputDS["x"], v)(
        add_1.construct(inputDS, tempDS),
        add_1.construct(tempDS, outputDS)

    );

    std::cout << call << std::endl;

    ComposableLower l;
    l.Lower(call);
    std::cout << l.Lower(call) << std::endl;

    call =
        For(outputDS["x"], v)(
            For(outputDS["x"], v1)(
                add_1.construct(inputDS, tempDS),
                add_1.construct(tempDS, outputDS)

                    ));
    std::cout << call << std::endl;
    ComposableLower l2;
    std::cout << l2.Lower(call) << std::endl;
}