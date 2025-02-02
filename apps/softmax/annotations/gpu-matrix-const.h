#pragma once

#include "annotations/abstract_function.h"

using namespace gern;

namespace annot {

template<bool Temp>
class MatrixGPU : public AbstractDataType {
public:
    MatrixGPU(const std::string &name)
        : name(name) {
    }
    std::string getName() const override {
        return name;
    }
    std::string getType() const override {
        if constexpr (temp) {
            return "auto";
        } else {
            return "MatrixGPU";
        }
    }

    std::vector<Variable> getFields() const{
                return {x, y, row, col};
                
    }
    FunctionSignature getAllocateFunction() const{
        return FunctionSignature {  
        };
    }
    FunctionSignature getFreeFunction() const{
        return FunctionSignature {  
        };
    }
    FunctionSignature getInsertFunction() const{
        return FunctionSignature {  
        };
    }
    FunctionSignature getQueryFunction() const{
        return FunctionSignature {  
        };
    }

private:
    std::string name;
     Variable x{"x"};
    Variable y{"y"};
    Variable row{"row"};
    Variable col{"col"};
    static constexpr bool temp = Temp;
};

};  // namespace annot