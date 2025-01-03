#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "functions/elementwise.h"
#include "functions/reduction.h"
#include "test-utils.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Lowering, SingleElemFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::add add_f;

    std::vector<Compose> c = {add_f(inputDS, outputDS)};
    Pipeline p(c);
    p.lower();
    p.generateCode();

    // std::cout << compose << std::endl;
}

TEST(Lowering, SingleReduceFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::reduction reduce_f;

    std::vector<Compose> c = {reduce_f(inputDS, outputDS)};
    Pipeline p(c);
    p.lower();
    std::cout << p << std::endl;

    // std::cout << compose << std::endl;
}

TEST(Lowering, MultiFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::add add_f;

    Compose compose{{add_f(inputDS, outputDS),
                     Compose(add_f(outputDS, outputDS))}};
    Pipeline p({compose});

    // Currently, only pipelines with one function call
    // can be lowered. This will (obviously) be removed
    // as I make progress!
    ASSERT_THROW(p.lower(), error::InternalError);
}