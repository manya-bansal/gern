#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

// TEST(LoweringCPU, SimpleNested) {
//     auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
//     auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
//     auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

//     annot::add_1 add_1;
//     Variable v1("v1");
//     Variable v2("v2");
//     Variable step("step");

//     Compose functions({
//         For(outputDS["size"], v2,
//             Tile(
//                 For(tempDS["size"], v1,
//                     Tile(add_1(inputDS, tempDS))),
//                 add_1(tempDS, outputDS))),
//     });

//     // std::cout << functions << std::endl;

//     Runner run(functions);
//     run.compile(test::cpuRunner("array"));

//     int64_t inner_step = 1;
//     int64_t outer_step = 5;

//     impl::ArrayCPU a(10);
//     a.ascending();
//     impl::ArrayCPU b(10);

//     run.evaluate({
//         {v1.getName(), &inner_step},
//         {v2.getName(), &outer_step},
//         {inputDS.getName(), &a},
//         {outputDS.getName(), &b},
//     });

//     std::cout << a << std::endl;
//     std::cout << b << std::endl;
//     // Make sure we got the correct answer.
//     for (int i = 0; i < 10; i++) {
//         ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
//     }

//     a.destroy();
//     b.destroy();
// }