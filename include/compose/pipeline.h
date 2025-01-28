// #pragma once

// #include "annotations/abstract_function.h"
// #include "compose/compose.h"
//
// #include "utils/uncopyable.h"

// namespace gern {

// using ComputeFunctionCallPtr = const ComputeFunctionCall *;

// // The pipeline actually holds the lowered
// // nodes, and helps us nest pipelines.
// class Pipeline {

// public:
//     Pipeline(std::vector<Compose> compose, bool fuse = true);
//     Pipeline(Compose compose);

//     std::vector<Compose> getFuncs() const;
//     std::vector<Compose> getRWFuncs() const;
//     // Returns the FunctionSignature call that produces a particular output.
//     std::set<AbstractDataTypePtr> getInputs() const;
//     std::set<ComputeFunctionCallPtr> getConsumerFunctions(AbstractDataTypePtr) const;
//     ComputeFunctionCallPtr getProducerFunction(AbstractDataTypePtr ds) const;
//     std::vector<AbstractDataTypePtr> getAllOutputs() const;
//     std::set<AbstractDataTypePtr> getIntermediates() const;
//     bool isTemplateArg(Variable v) const;
//     bool isIntermediate(AbstractDataTypePtr d) const;

//     Pattern getAnnotation() const;
//     AbstractDataTypePtr getOutput() const;
//     std::vector<Assign> getDefinitions() const;
//     std::set<Compose> getConsumers(AbstractDataTypePtr ds) const;

//     void accept(CompositionVisitorStrict *) const;

//     std::set<AbstractDataTypePtr> getAllWriteDataStruct() const;  // This gathers all the data-structures written to in the pipeline.
//     std::set<AbstractDataTypePtr> getAllReadDataStruct() const;   // This gathers all the data-structures written to in the pipeline.

// private:
//     void check_if_legal(std::vector<Compose> compose);  // Initializes private vars, and ensures that the user has constructed a valid pipeline.
//     void constructAnnotation();
//     Consumes generateConsumesIntervals(Compose c, std::vector<SubsetObj> input_subsets) const;
//     Pattern generateProducesIntervals(Compose, Computes) const;
//     std::vector<Compose> compose;
//     std::vector<Compose> rw_compose;
//     AbstractDataTypePtr true_output;
//     std::vector<AbstractDataTypePtr> all_outputs;
//     std::set<AbstractDataTypePtr> intermediates_set;
//     std::set<AbstractDataTypePtr> inputs;
//     std::set<Variable> const_exprs;
//     std::vector<Assign> definitions;
//     Pattern annotation;
//     // The last function of a child's pipeline, needs to be refreshed.
//     // The parent pipeline and the child pipeline should refer to different
//     // fields.
//     std::map<ComputeFunctionCallPtr, Compose> fresh_calls;
//     bool fuse = false;
// };

// std::ostream &operator<<(std::ostream &os, const Pipeline &p);

// struct PipelineNode : public CompositionObject {
//     PipelineNode(Pipeline p)
//         : p(p) {
//     }

//     void accept(CompositionVisitorStrict *) const;
//     Pattern getAnnotation() const;
//     bool isTemplateArg(Variable v) const;
//     Pipeline p;
// };

// // Defining an abstract data class that we can use to define query and free node.
// class DummyDS : public AbstractDataType {
// public:
//     DummyDS(const std::string &name,
//                const std::string &type,
//                const std::vector<Variable> &fields,
//                const FunctionSignature &allocate,
//                const FunctionSignature &free,
//                const FunctionSignature &insert,
//                const FunctionSignature &query,
//                const bool &to_free)
//         : name(name), type(type), fields(fields),
//           allocate(allocate), free(free),
//           insert(insert), query(query),
//           to_free(to_free) {
//     }

//     virtual std::string getName() const override {
//         return name;
//     }

//     virtual std::string getType() const override {
//         return type;
//     }

//     std::vector<Variable> getFields() const override {
//         return fields;
//     }
//     FunctionSignature getAllocateFunction() const override {
//         return allocate;
//     }
//     FunctionSignature getFreeFunction() const override {
//         return free;
//     }
//     FunctionSignature getInsertFunction() const override {
//         return insert;
//     }
//     FunctionSignature getQueryFunction() const override {
//         return query;
//     }

//     // Tracks whether any of the queries need to be free,
//     // or if they are actually returning views.
//     bool freeQuery() const override {
//         return to_free;
//     }

//     static AbstractDataTypePtr make(const std::string &name,
//                                     const std::string &type,
//                                     AbstractDataTypePtr ds) {
//         return AbstractDataTypePtr(new const DummyDS(name, type,
//                                                         ds.ptr->getFields(),
//                                                         ds.ptr->getAllocateFunction(),
//                                                         ds.ptr->getFreeFunction(),
//                                                         ds.ptr->getInsertFunction(),
//                                                         ds.ptr->getQueryFunction(),
//                                                         ds.ptr->freeQuery()));
//     }

//     static AbstractDataTypePtr make(const std::string &name,
//                                     AbstractDataTypePtr ds) {
//         return make(name, ds.getType(), ds);
//     }

// private:
//     std::string name;
//     std::string type;
//     std::vector<Variable> fields;
//     FunctionSignature allocate;
//     FunctionSignature free;
//     FunctionSignature insert;
//     FunctionSignature query;
//     bool to_free;
// };

// }  // namespace gern