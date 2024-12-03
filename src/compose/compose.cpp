#include "compose/compose.h"

namespace gern {

Compose::Compose(const std::vector<FunctionCall> &funcs)
    : funcs(funcs) {
}

// I am pulling concretize out since I expect
// certain scheduling commands to have loops be
// explicitly instantiated. If this is not the case,
// I can pull concretize into the constructor :)
void Compose::concretize() {
    concrete = true;
}

bool Compose::concretized() const {
    return concrete;
}

const std::vector<FunctionCall> &Compose::getFunctions() const {
    return funcs;
}

std::ostream &operator<<(std::ostream &os, const Compose &compose) {
    for (const auto &f : compose.getFunctions()) {
        os << f << "\n";
    }
    return os;
}

}  // namespace gern