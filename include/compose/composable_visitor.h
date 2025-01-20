#pragma once

#include "compose/composable.h"

namespace gern {

class ComputationNode;
class TiledComputationNode;

class ComposableVisitorStrict {
public:
    virtual void visit(Composable);
    virtual void visit(const ComputationNode *) = 0;
    virtual void visit(const TiledComputationNode *) = 0;
};

}  // namespace gern