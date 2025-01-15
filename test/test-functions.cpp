#include "annotations/visitor.h"
#include "compose/compose.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Functions, CallFunctions) {
    annot::add add_f;
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    // Call with incorrect number of arguments.
    ASSERT_THROW(add_f(), error::UserError);
    ASSERT_THROW(add_f(inputDS), error::UserError);
    // Call with more than required.
    ASSERT_THROW(add_f(inputDS, inputDS, inputDS), error::UserError);
    // Try calling with an undefined argument.
    ASSERT_THROW(add_f(Argument(), Argument()), error::UserError);

    // Try calling with correct arguments.
    // The fact that we are overwriting the input is caught in
    // the pipeline step, as opposed to this step.
    ASSERT_NO_THROW(add_f(inputDS, inputDS));

    annot::addWithSize add_with_size;
    // Call with incorrect type of arg.
    ASSERT_THROW(add_with_size(inputDS, Variable{"x"}, inputDS), error::UserError);
    // Call with correct type of arg.
    ASSERT_NO_THROW(add_with_size(inputDS, inputDS, Variable{"x"}));
}

TEST(Functions, Binding) {
    // Try to incorrectly bind variables to the grid.
    annot::add add_f;
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    Variable x("x");
    ASSERT_THROW((add_f[{
                     {"x", x.bindToGrid(Grid::Property::BLOCK_DIM_X)},
                 }](inputDS, inputDS)),
                 error::UserError);
}

TEST(Functions, ReplaceWithFreshVars) {
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    AbstractDataTypePtr outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    AbstractDataTypePtr output_2_DS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_2"));

    annot::add add_f;

    Compose c1 = add_f(inputDS, outputDS);
    Compose c2 = add_f(outputDS, output_2_DS);

    ComputeFunctionCallPtr concreteCall1 = to<ComputeFunctionCall>(c1);
    ComputeFunctionCallPtr concreteCall2 = to<ComputeFunctionCall>(c2);

    // Test that all variables were replaced
    std::set<Variable> abstract_vars = getVariables(add_f.getAnnotation());
    std::set<Variable> concrete_vars_1 = getVariables(concreteCall1->getAnnotation());
    std::set<Variable> concrete_vars_2 = getVariables(concreteCall2->getAnnotation());

    // All the variables should now be different.
    ASSERT_TRUE(test::areDisjoint(concrete_vars_1, abstract_vars));
    ASSERT_TRUE(test::areDisjoint(concrete_vars_2, abstract_vars));
    ASSERT_TRUE(test::areDisjoint(concrete_vars_2, concrete_vars_1));

    AbstractDataTypePtr output_1 = concreteCall1->getOutput();
    ASSERT_TRUE(output_1 == outputDS);
    AbstractDataTypePtr output_2 = concreteCall2->getOutput();
    ASSERT_TRUE(output_2 == output_2_DS);

    std::set<AbstractDataTypePtr> all_inputs = concreteCall1->getInputs();
    ASSERT_TRUE(all_inputs.size() == 1);
    ASSERT_TRUE(*(all_inputs.begin()) == inputDS);

    std::set<AbstractDataTypePtr> all_inputs_2 = concreteCall2->getInputs();
    ASSERT_TRUE(all_inputs_2.size() == 1);
    ASSERT_TRUE(*(all_inputs_2.begin()) == outputDS);
}