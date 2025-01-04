#ifndef GERN_REDUCTION_FUNCTION_H
#define GERN_REDUCTION_FUNCTION_H

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace test {

// This is perhaps a contrived example, but it exists to
// exercise the ability to add for loops inside
// the compute annotation.
class reduction : public AbstractFunction {
public:
    reduction()
        : input(std::make_shared<dummy::TestDS>("input")),
          output(std::make_shared<dummy::TestDS>("output")) {
    }
    std::string getName() {
        return "gern::lib::add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable end("end");

        return For(x = Expr(0), x < end, x = x + step,
                   Computes(
                       Produces(
                           Subset(output, {x, 1})),
                       Consumes(
                           For(r = Expr(0), r < end, r = r + 1,
                               Subsets{
                                   Subset(input, {r, 1})}))));
    }

    std::vector<Argument> getArguments() {
        return {Argument(input), Argument(output)};
    }

    std::vector<std::string> getHeader() {
        return {
            "array_lib.h",
        };
    }

    std::vector<std::string> getIncludeFlags() {
        return {
            std::string(GERN_ROOT_DIR) + "/test/library/array/",
        };
    }

private:
    std::shared_ptr<dummy::TestDS> input;
    std::shared_ptr<dummy::TestDS> output;
};

}  // namespace test
}  // namespace gern

#endif