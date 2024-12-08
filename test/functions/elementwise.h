#ifndef GERN_ELEMENTWISE_FUNCTION_H
#define GERN_ELEMENTWISE_FUNCTION_H

#include "annotations/abstract_function.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace test {

class add : public AbstractFunction {
public:
    add()
        : input(std::make_shared<dummy::TestDS>("input")),
          output(std::make_shared<dummy::TestDS>("output")) {
    }
    std::string getName() {
        return "add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Variable step("step");
        Variable end("end");

        return For(x.get_from_grid(), 0, end, step,
                   Computes(
                       Produces(
                           Subset(output, {x})),
                       Consumes(
                           Subset(input, {x}))));
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