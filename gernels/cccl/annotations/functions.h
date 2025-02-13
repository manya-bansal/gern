#include "annotations/abstract_function.h"
#include "compose/composable.h"

using namespace gern;

namespace annot {

class global_sum : public AbstractFunction {
public:
    global_sum() = default;

private:
    AbstractDataType input;
    AbstractDataType output;
};

}  // namespace annot