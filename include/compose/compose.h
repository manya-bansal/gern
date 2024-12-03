#ifndef GERN_COMPOSE_H
#define GERN_COMPOSE_H

#include "annotations/abstract_function.h"

namespace gern {
/**
 * @brief This class describes composition of functions
 *         that Gern will generate code for.
 *
 */
class Compose {
public:
    Compose(const std::vector<FunctionCall> &funcs);
    void concretize();
    bool concretized() const;

    // Returning a const to ensure that the function vector
    // cannot be changed externally.
    const std::vector<FunctionCall> &getFunctions() const;

private:
    std::vector<FunctionCall> funcs;
    bool concrete = false;
};

std::ostream &operator<<(std::ostream &os, const Compose &);

}  // namespace gern

#endif