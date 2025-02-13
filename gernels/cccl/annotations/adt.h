#include "annotations/composable.h"
#include "annotations/data_dependency_language.h"

using namespace gern;

namespace annot {

class FloatPtr : public AbstractDataType {
public:
    FloatPtr(const std::string &name)
        : name(name) {
    }

    std::string getType() const override {
        return "float*";
    }

    std::string getName() const override {
        return name;
    }

    // No fields to return
    std::vector<Variable> getFields() const override {
        return {};
    }

    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{};
    }

    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{};
    }

    bool insertQuery() const override {
        return false;
    }

    bool freeQuery() const override {
        return false;
    }

private:
    std::string name;
};

}  // namespace annot