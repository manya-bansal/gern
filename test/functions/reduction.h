#ifndef GERN_REDUCTION_FUNCTION_H
#define GERN_REDUCTION_FUNCTION_H

#include "annotations/abstract_function.h"
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
        return "reduction";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable end("end");

        return For(x, 0, end, step,
                   Computes(
                       Produces(
                           Subset(output, {x})),
                       Consumes(
                           For(r, 0, 8, 1,
                               Subsets{
                                   Subset(input, {x * 8 + r})}))));
    }

    std::vector<Argument> getArguments() {
        return {Argument(input), Argument(output)};
    }

private:
    std::shared_ptr<dummy::TestDS> input;
    std::shared_ptr<dummy::TestDS> output;
};

}  // namespace test
}  // namespace gern

#endif