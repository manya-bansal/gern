#include "annotations/rewriter_helpers.h"
#include "annotations/visitor.h"
#include "compose/compose.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Arguments, isSameType) {
    Variable x("x");
    Variable y("y");

    ASSERT_FALSE(Argument(x + y).isSameTypeAs(x));
    ASSERT_TRUE(Argument(x + y).isSameTypeAs(Argument(x + y)));
}

TEST(Functions, CallFunctions) {
    annot::add_1 add_1;
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    AbstractDataTypePtr outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    // Call with incorrect number of arguments.
    ASSERT_THROW(add_1(), error::UserError);
    ASSERT_THROW(add_1(inputDS), error::UserError);
    // Call with more than required.
    ASSERT_THROW(add_1(inputDS, inputDS, outputDS), error::UserError);
    // Try calling with an undefined argument.
    ASSERT_THROW(add_1(Argument(), Argument()), error::UserError);

    // Try to overwrite the input.
    ASSERT_THROW(add_1(inputDS, inputDS), error::UserError);

    annot::addWithSize add_with_size;
    // Call with incorrect type of arg.
    ASSERT_THROW(add_with_size(inputDS, Variable{"x"}, outputDS), error::UserError);
    // Call with correct type of arg.
    ASSERT_NO_THROW(add_with_size(inputDS, outputDS, Variable{"x"}));
}

TEST(Functions, ReplaceWithFreshVars) {
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    AbstractDataTypePtr outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    AbstractDataTypePtr output_2_DS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_2"));

    annot::add_1 add_1;

    Composable c1 = add_1(inputDS, outputDS);
    Composable c2 = add_1(outputDS, output_2_DS);

    // Test that all variables were replaced
    std::set<Variable> abstract_vars = getVariables(add_1.getAnnotation().getPattern());
    std::set<Variable> concrete_vars_1 = getVariables(c1.getAnnotation().getPattern());
    std::set<Variable> concrete_vars_2 = getVariables(c2.getAnnotation().getPattern());

    // All the variables should now be different.
    ASSERT_TRUE(test::areDisjoint(concrete_vars_1, abstract_vars));
    ASSERT_TRUE(test::areDisjoint(concrete_vars_2, abstract_vars));
    ASSERT_TRUE(test::areDisjoint(concrete_vars_2, concrete_vars_1));

    AbstractDataTypePtr output_1 = c1.getAnnotation().getPattern().getOutput().getDS();
    ASSERT_TRUE(output_1 == outputDS);
    AbstractDataTypePtr output_2 = c2.getAnnotation().getPattern().getOutput().getDS();
    ASSERT_TRUE(output_2 == output_2_DS);

    std::vector<SubsetObj> all_inputs = c1.getAnnotation().getPattern().getInputs();
    ASSERT_TRUE(all_inputs.size() == 1);
    ASSERT_TRUE(all_inputs.begin()->getDS() == inputDS);

    std::vector<SubsetObj> all_inputs_2 = c2.getAnnotation().getPattern().getInputs();
    ASSERT_TRUE(all_inputs_2.size() == 1);
    ASSERT_TRUE(all_inputs_2.begin()->getDS() == outputDS);
}