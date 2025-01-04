#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "functions/elementwise.h"
#include "functions/reduction.h"
#include "library/array/array_lib.h"

#include "config.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Lowering, SingleElemFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");

    test::add add_f;
    Variable v("v");

    std::vector<Compose> c = {add_f(inputDS, outputDS, v)};
    Pipeline p(c);

    p.lower();
    void *fp = p.evaluate("-std=c++11 -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/");

    void (*f)(void **) = (void (*)(void **))fp;

    // Actually use the array
    TestArray a(10);
    a.vvals(2.0f);
    TestArray b(10);
    b.vvals(3.0f);
    int64_t var = 10;

    void *args[] = {&a, &b, &var};
    f(args);

    for (int i = 0; i < 10; i++) {
        std::cout << b.data[i] << std::endl;
        std::cout << a.data[i] << std::endl;
    }
}

TEST(Lowering, SingleReduceFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::reduction reduce_f;

    std::vector<Compose> c = {reduce_f(inputDS, outputDS)};
    Pipeline p(c);

    p.lower();
    // p.evaluate();
}

TEST(Lowering, MultiFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::add add_f;
    Variable v("v");

    Compose compose{{add_f(inputDS, outputDS, v),
                     Compose(add_f(outputDS, outputDS, v))}};
    Pipeline p({compose});

    // Currently, only pipelines with one function call
    // can be lowered. This will (obviously) be removed
    // as I make progress!
    ASSERT_THROW(p.lower(), error::InternalError);
}