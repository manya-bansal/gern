#include "annotations/abstract_nodes.h"

namespace gern {

std::ostream &operator<<(std::ostream &os, const AbstractDataType &ads) {
    os << ads.getName();
    return os;
}

}  // namespace gern