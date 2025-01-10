#include "compose/compose.h"
#include "compose/pipeline.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(PipelineTest, ReuseOutput) {
    Argument inputDS = Argument(AbstractDataTypePtr(new const annot::ArrayCPU("input_con")));
    Argument outputDS = Argument(AbstractDataTypePtr(new const annot::ArrayCPU("output_con")));
    Argument tempDS = Argument(AbstractDataTypePtr(new const annot::ArrayCPU("output_con")));

    annot::add add_f;
    Pipeline p({
        add_f(inputDS, tempDS),
        add_f(tempDS, outputDS),
    });

    // A pipeline can only assign to fresh outputs
    // each time.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     add_f(outputDS, outputDS),
                 }),
                 error::UserError);

    // A pipeline can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     {
                         add_f(outputDS, outputDS),
                     },
                 }),
                 error::UserError);

    // A pipeline can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_THROW(Pipeline p({
                     Compose(
                         add_f(outputDS, outputDS)),
                     add_f(inputDS, outputDS),
                 }),
                 error::UserError);
}

TEST(PipelineTest, NoReuseOutput) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add add_f;

    ASSERT_NO_THROW(Pipeline p({
        add_f(inputDS, outputDS),
        add_f(outputDS, outputDS_new),
    }));

    // A pipeline can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_NO_THROW(Pipeline p({
        add_f(inputDS, outputDS),
        {
            add_f(outputDS, outputDS_new),
        },
    }));
}

TEST(PipelineTest, ExtraOutput) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add add_f;

    // Gern does not allow computing outputs that are not
    // eventually used in the rest of the pipeline.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     add_f(inputDS, outputDS_new),
                 }),
                 error::UserError);
    // Catch in nested pipeline too.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     {
                         add_f(inputDS, outputDS_new),
                     },
                 }),
                 error::UserError);

    // Catch in nested pipeline that appears first.
    ASSERT_THROW(Pipeline p({
                     {
                         add_f(inputDS, outputDS_new),
                     },
                     add_f(inputDS, outputDS),
                 }),
                 error::UserError);

    ASSERT_THROW(Pipeline p({
                     Compose({
                         Compose({
                             Compose({
                                 add_f(inputDS, outputDS_new),
                             }),
                         }),
                     }),

                     add_f(inputDS, outputDS),
                 }),
                 error::UserError);
}

TEST(PipelineTest, AssignInput) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add add_f;
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     add_f(outputDS, inputDS),
                 }),
                 error::UserError);
}

TEST(PipelineTest, getConsumerFuncs) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add add_f;

    // Each invocation creates a new object,
    // and we need a reference to the object to actually
    // test!
    auto add_1 = add_f(inputDS, outputDS);
    auto add_2 = add_f(outputDS, outputDS_new);
    Pipeline p({
        add_1,
        add_2,
    });

    auto funcs = p.getConsumerFunctions(outputDS);
    ASSERT_TRUE(funcs.size() == 1);
    ASSERT_TRUE(funcs.contains(add_2));

    Pipeline p_nested({add_1,
                       {
                           Compose(add_2),
                       }});
    funcs = p_nested.getConsumerFunctions(outputDS);
    ASSERT_TRUE(funcs.size() == 1);
    ASSERT_TRUE(funcs.contains(add_2));
    funcs = p_nested.getConsumerFunctions(outputDS_new);
    ASSERT_TRUE(funcs.size() == 0);

    Pipeline p_nested_2({
        Compose(
            Compose(
                Compose(
                    add_1))),
        add_2,
    });

    funcs = p_nested_2.getConsumerFunctions(outputDS);
    ASSERT_TRUE(funcs.size() == 1);
    ASSERT_TRUE(funcs.contains(add_2));
    funcs = p_nested_2.getConsumerFunctions(outputDS_new);
    ASSERT_TRUE(funcs.size() == 0);
}