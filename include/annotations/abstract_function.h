#ifndef GERN_ABSTRACT_FUNCTION_H
#define GERN_ABSTRACT_FUNCTION_H

#include "annotations/data_dependency_language.h"
#include "utils/error.h"
#include "utils/uncopyable.h"
#include <any>
#include <string>

namespace gern {

class FunctionCall;
typedef int Argument;
/**
 * @brief To add a function to gern, the user needs to
 *        define an instance of the AbstractFunction class.
 *
 * The class consists of  x virtual function;
 *
 */
class AbstractFunction {
public:
    AbstractFunction() = default;
    virtual ~AbstractFunction() = default;

    virtual std::string getName() = 0;
    virtual Stmt getAnnotation() = 0;
    virtual std::vector<Argument> getArguments() = 0;

    template<typename FirstT, typename... Next>
    std::shared_ptr<FunctionCall> operator()(FirstT first, Next... remaining) {
        std::vector<Argument> arguments;
        return std::make_shared<FunctionCall>(getName(), getAnnotation(), arguments);
    }

    std::shared_ptr<FunctionCall> operator()() {
        if (getArguments().size() != 0) {
            throw error::UserError("Called function " + getName() + " with 0 arguments");
        }
        return std::make_shared<FunctionCall>(getName(),
                                              getAnnotation(),
                                              std::vector<Argument>());
    }
};

class FunctionCall {
public:
    FunctionCall() = delete;
    FunctionCall(const std::string &name,
                 Stmt annotation,
                 std::vector<Argument> arguments)
        : name(name), annotation(annotation),
          arguments(arguments) {
    }
    std::string getName() const {
        return name;
    }
    Stmt getAnnotation() const {
        return annotation;
    }

private:
    std::string name;
    Stmt annotation;
    std::vector<Argument> arguments;
};

}  // namespace gern
#endif