#ifndef GERN_ELEMENTWISE_FUNCTION_H
#define GERN_ELEMENTWISE_FUNCTION_H

#include "annotations/abstract_function.h"
#include "config.h"
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
        Expr step(1);
        Variable end("end");

        return For(x = Expr(0), x < end, x = (x + step),
                   Computes(
                       Produces(
                           Subset(output, {x, step})),
                       Consumes(
                           Subset(input, {x, step}))));
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