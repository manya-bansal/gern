#include "annotations/visitor.h"
#include "compose/compose.h"
#include "functions/elementwise.h"
#include "test-utils.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Functions, SingleFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");

    test::add add_f;
    FunctionCall concreteCall1 = add_f(inputDS, outputDS);
    FunctionCall concreteCall2 = add_f(inputDS, outputDS);

    // Test that all variables were replaced
    std::set<Variable> abstract_vars = getVariables(add_f.getAnnotation());
    std::set<Variable> concrete_vars_1 = getVariables(concreteCall1.getAnnotation());
    std::set<Variable> concrete_vars_2 = getVariables(concreteCall2.getAnnotation());

    ASSERT_TRUE(areDisjoint(concrete_vars_1, abstract_vars));
    ASSERT_TRUE(areDisjoint(concrete_vars_2, abstract_vars));
    ASSERT_TRUE(areDisjoint(concrete_vars_2, concrete_vars_1));
}