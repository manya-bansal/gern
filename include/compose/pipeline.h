#pragma once

#include "annotations/abstract_function.h"
#include "compose/compose.h"
#include "compose/compose_visitor.h"
#include "utils/uncopyable.h"

namespace gern {

using ComputeFunctionCallPtr = const ComputeFunctionCall *;

// The pipeline actually holds the lowered
// nodes, and helps us nest pipelines.
class Pipeline {

public:
    Pipeline(std::vector<Compose> compose, bool fuse = true);
    Pipeline(Compose compose);

    std::vector<Compose> getFuncs() const;
    // Returns the FunctionSignature call that produces a particular output.
    std::set<AbstractDataTypePtr> getInputs() const;
    AbstractDataTypePtr getOutput() const;
    std::set<ComputeFunctionCallPtr> getConsumerFunctions(AbstractDataTypePtr) const;
    ComputeFunctionCallPtr getProducerFunction(AbstractDataTypePtr ds) const;
    std::vector<AbstractDataTypePtr> getAllOutputs() const;
    std::set<AbstractDataTypePtr> getIntermediates() const;

    void accept(CompositionVisitorStrict *) const;

    std::set<AbstractDataTypePtr> getAllWriteDataStruct() const;  // This gathers all the data-structures written to in the pipeline.
    std::set<AbstractDataTypePtr> getAllReadDataStruct() const;   // This gathers all the data-structures written to in the pipeline.

private:
    void init(std::vector<Compose> compose);  // Initializes private vars, and ensures that the user has constructed a valid pipeline.
    std::vector<Compose> compose;
    AbstractDataTypePtr true_output;
    std::vector<AbstractDataTypePtr> all_outputs;
    std::set<AbstractDataTypePtr> intermediates_set;
    std::set<AbstractDataTypePtr> inputs;
    // The last function of a child's pipeline, needs to be refreshed.
    // The parent pipeline and the child pipeline should refer to different
    // fields.
    std::map<ComputeFunctionCallPtr, Compose> fresh_calls;
    bool fuse = false;
};

std::ostream &operator<<(std::ostream &os, const Pipeline &p);

struct PipelineNode : public CompositionObject {
    PipelineNode(Pipeline p)
        : p(p) {
    }

    void accept(CompositionVisitorStrict *) const;
    Pipeline p;
};

}  // namespace gern