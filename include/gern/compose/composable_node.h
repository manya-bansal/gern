#pragma once

#include "annotations/data_dependency_language.h"
#include "utils/uncopyable.h"

namespace gern {

class ComposableVisitorStrict;

/**
 * @brief ComposableNode is an object that Gern understands
 *        how to compose.
 *
 */
class ComposableNode : public util::Manageable<ComposableNode>,
                       public util::Uncopyable {
public:
    ComposableNode() = default;
    virtual ~ComposableNode() = default;
    virtual void accept(ComposableVisitorStrict *) const = 0;
    virtual Annotation getAnnotation() const = 0;
};

}  // namespace gern