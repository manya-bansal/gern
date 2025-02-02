#pragma once
#include "annotations/abstract_function.h"
#include "compose/composable.h"

using namespace gern;

class Hello : public AbstractFunction {
public:
    Annotation getAnnotation() override {
        return Annotation();
    }

    std::vector<std::string> getHeader() override {
        return {
            "hello.h",
        };
    }

    FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "hello";
        return f;
    }
};