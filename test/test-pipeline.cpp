#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(PipelineTest, ReuseOutput) {
    AbstractDataTypePtr inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    AbstractDataTypePtr outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_val"));
    AbstractDataTypePtr tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("tempDS"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    ASSERT_NO_THROW(Composable());
    ASSERT_NO_THROW(Composable({
        add_1(inputDS, tempDS),
        add_1(tempDS, outputDS),
    }));

    // Can only assign to fresh outputs
    // each time.
    // ASSERT_THROW(Composable(
    //                  std::vector<Composable>{
    //                      add_1(inputDS, outputDS),
    //                      add_1(outputDS, outputDS),
    //                  }),
    //              error::UserError);

    // Can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_THROW(Composable({
                     add_1(inputDS, outputDS),
                     Tile(outputDS["size"], v)(
                         add_1(inputDS, outputDS)),
                 }),
                 error::UserError);

    // A pipeline can only assign to fresh outputs
    // each time. Should complain even if nested.
    ASSERT_THROW(Tile(outputDS["size"], v)(
                     Tile(outputDS["size"], v1)(
                         add_1(outputDS, outputDS)),
                     add_1(inputDS, outputDS)),
                 error::UserError);

    // Try to assign to an output that has been read by the child.
    ASSERT_THROW(Tile(outputDS["size"], v)(
                     Tile(outputDS["size"], v1)(
                         add_1(inputDS, outputDS)),
                     add_1(outputDS, inputDS)),
                 error::UserError);

    // Assign to input that has been read by parent.
    ASSERT_THROW(Tile(outputDS["size"], v)(
                     add_1(inputDS, outputDS),
                     Tile(outputDS["size"], v1)(
                         add_1(outputDS, inputDS))),
                 error::UserError);
}

TEST(PipelineTest, NoReuseOutput) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    ASSERT_NO_THROW(Composable({
        add_1(inputDS, outputDS),
        add_1(outputDS, outputDS_new),
    }));

    ASSERT_NO_THROW(Tile(outputDS_new["size"], v)(
        add_1(inputDS, outputDS),
        Tile(outputDS_new["size"], v1)(
            add_1(outputDS, outputDS_new))));
}

TEST(PipelineTest, IntermediateVisibility) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    auto outputDS_vis = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_vis"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");
    // Try to read a child's intermediate.
    ASSERT_THROW(Composable({
                     Tile(outputDS_new["size"], v)(
                         add_1(inputDS, outputDS),
                         add_1(outputDS, outputDS_new)),
                     add_1(outputDS, outputDS_vis),
                 }),
                 error::UserError);
    // Try to write to a child's intermediate.
    ASSERT_THROW(Composable({
                     Tile(outputDS_new["size"], v)(
                         add_1(inputDS, outputDS),
                         add_1(outputDS, outputDS_new)),
                     add_1(outputDS_new, outputDS),
                 }),
                 error::UserError);

    // Should be okay now
    ASSERT_NO_THROW(Composable({
        Composable({
            add_1(inputDS, outputDS),
            add_1(outputDS, outputDS_new),
        }),
        add_1(outputDS_new, outputDS_vis),
    }));
}

TEST(PipelineTest, ExtraOutput) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto outputDS_new = AbstractDataTypePtr(new const annot::ArrayCPU("output_con_new"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Gern does not allow computing outputs that are not
    // eventually used in the rest of the pipeline.
    ASSERT_THROW(Composable({
                     add_1(inputDS, outputDS),
                     add_1(inputDS, outputDS_new),
                 }),
                 error::UserError);
    // Catch in nested pipeline too.
    ASSERT_THROW(Composable({
                     add_1(inputDS, outputDS),
                     Tile(outputDS_new["size"], v)(
                         add_1(inputDS, outputDS_new)),
                 }),
                 error::UserError);

    // Catch in nested pipeline that appears first.
    ASSERT_THROW(Composable({
                     Tile(outputDS_new["size"], v)(
                         add_1(inputDS, outputDS_new)),
                     add_1(inputDS, outputDS),
                 }),
                 error::UserError);

    ASSERT_THROW(Composable({
                     Tile(outputDS_new["size"], v)(
                         Tile(outputDS_new["size"], v)(
                             Tile(outputDS_new["size"], v1)(
                                 add_1(inputDS, outputDS_new)))),
                     add_1(inputDS, outputDS),
                 }),
                 error::UserError);
}
