#include "compose/compose.h"
#include "compose/pipeline.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(PipelineTest, ReuseOutput) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

    annot::add add_f;

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

    auto outputDS2 = std::make_shared<const annot::ArrayCPU>("output_con");
    // A pipeline can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     Compose({
                         add_f(outputDS, outputDS2),
                         add_f(outputDS, outputDS2),
                     }),
                 }),
                 error::UserError);
}

TEST(PipelineTest, NoReuseOutput) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    auto outputDS_new = std::make_shared<const annot::ArrayCPU>("output_con_new");

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
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    auto outputDS_new = std::make_shared<const annot::ArrayCPU>("output_con_new");

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
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    auto outputDS_new = std::make_shared<const annot::ArrayCPU>("output_con_new");

    annot::add add_f;
    // Gern does not allow assigning to "true" inputs.
    ASSERT_THROW(Pipeline p({
                     add_f(inputDS, outputDS),
                     add_f(outputDS, inputDS),
                 }),
                 error::UserError);
}

TEST(PipelineTest, getInputs) {

    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    auto outputDS_new = std::make_shared<const annot::ArrayCPU>("output_con_new");

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
    // ASSERT_TRUE(p.getProducerFunc(outputDS) == add_1);
    // ASSERT_TRUE(p.getProducerFunc(outputDS_new) == add_2);

    Pipeline p_nested({add_1,
                       {
                           {add_2},
                       }});
    // ASSERT_TRUE(p_nested.getProducerFunc(outputDS) == add_1);
    // ASSERT_TRUE(p_nested.getProducerFunc(outputDS_new) == add_2);
    // // We should not be able to find a function.
    // ASSERT_TRUE(p_nested.getProducerFunc(inputDS) == nullptr);

    Pipeline p_nested_2({
        Compose({
            Compose({
                Compose({
                    add_1,
                }),
            }),
        }),
        add_2,
    });

    // ASSERT_TRUE(p_nested.getProducerFunc(outputDS) == add_1);
    // ASSERT_TRUE(p_nested.getProducerFunc(outputDS_new) == add_2);
    // ASSERT_TRUE(p_nested_2.getProducerFunc(inputDS) == nullptr);
}