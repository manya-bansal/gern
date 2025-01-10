#include "annotations/visitor.h"
#include "compose/compose.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Functions, SingleFunction) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    auto output_2_DS = std::make_shared<const annot::ArrayCPU>("output_con_2");

    annot::add add_f;
    const ComputeFunctionCall *concreteCall1 = add_f(inputDS, outputDS);
    const ComputeFunctionCall *concreteCall2 = add_f(outputDS, output_2_DS);

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