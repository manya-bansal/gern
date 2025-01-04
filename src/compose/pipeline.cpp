#include "compose/pipeline.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include "codegen/codegen.h"
#include "compose/compose.h"
#include "compose/pipeline_visitor.h"
#include "utils/debug.h"
#include "utils/error.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>

namespace gern {

void LowerIR::accept(PipelineVisitor *v) const {
    if (!defined()) {
        return;
    }
    ptr->accept(v);
}

std::ostream &operator<<(std::ostream &os, const LowerIR &n) {
    PipelinePrinter print(os, 0);
    print.visit(n);
    return os;
}

Pipeline &Pipeline::at_device() {
    device = true;
    return *this;
}

Pipeline &Pipeline::at_host() {
    device = false;
    return *this;
}

bool Pipeline::is_at_device() const {
    return device;
}

void Pipeline::lower() {

    // Convert it into a ComposeVec
    auto compose_vec = Compose(new const ComposeVec(compose));

    if (compose_vec.numFuncs() != 1) {
        throw error::InternalError("Lowering is only "
                                   "implemented for one function!");
    }

    visit(compose_vec);
}

void Pipeline::compile(std::string compile_flags) {

    codegen::CodeGenerator cg;
    codegen::CGStmt code = cg.generate_code(*this);

    std::string prefix = "/tmp/";
    std::string file = prefix + "test.cpp";
    // Write the code to a file.
    std::ofstream outFile(file);
    outFile << code;
    outFile.close();

    std::string shared_obj = prefix + getUniqueName("libGern") + ".so";
    std::string cmd = "g++ " + compile_flags + " -shared -o " + shared_obj + " " + file + " 2>&1";

    // Compile the code.
    int runStatus = std::system(cmd.data());
    if (runStatus != 0) {
        throw error::UserError("Compilation Failed");
    }

    void *handle = dlopen(shared_obj.data(), RTLD_LAZY);
    if (!handle) {
        throw error::UserError("Error loading library: " + std::string(dlerror()));
    }

    void *func = dlsym(handle, cg.getHookName().data());
    if (!func) {
        throw error::UserError("Error loading function: " + std::string(dlerror()));
    }

    fp = (GernGenFuncPtr)func;
    argument_order = cg.getArgumentOrder();
    compiled = true;
}

void Pipeline::evaluate(std::map<std::string, void *> args) {
    if (!compiled) {
        this->compile();
    }

    size_t num_args = argument_order.size();
    if (args.size() != num_args) {
        throw error::UserError("All the arguments have not been passed! Expecting " + std::to_string(num_args) + " args");
    }
    // Now, fp has the function pointer,
    // and argument order contains the order
    // in which the arguments need to be set into
    // a void **.
    void **args_in_order = (void **)malloc(sizeof(void *) * num_args);
    int arg_num = 0;
    for (const auto &a : argument_order) {
        if (args.count(a) <= 0) {
            throw error::UserError("Argument " + a + "was not passed in");
        }
        args_in_order[arg_num] = args.at(a);
        arg_num++;
    }

    // Now, actually run the function.
    fp(args_in_order);
    // Free storage.
    free(args_in_order);
}

std::map<Variable, Expr> Pipeline::getVariableDefinitions() const {
    return variable_definitions;
}

std::vector<LowerIR> Pipeline::getIRNodes() const {
    return lowered;
}

std::ostream &operator<<(std::ostream &os, const Pipeline &p) {
    PipelinePrinter print(os, 0);
    print.visit(p);
    return os;
}

void Pipeline::visit(const FunctionCall *c) {

    std::set<AbstractDataTypePtr> inputs = c->getInputs();
    std::map<AbstractDataTypePtr, AbstractDataTypePtr> new_ds;
    std::vector<LowerIR> temp;

    for (const auto &in : inputs) {
        if (!isIntermediate(in)) {
            // Generate the query node.
            AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + in->getName(),
                                                                                   in->getType());
            new_ds[in] = queried;
            temp.push_back(new QueryNode(in,
                                         queried,
                                         generateMetaDataFields(in, c)));
        } else {
            // Generate an allocate node!
            throw error::InternalError("Unimplemented");
        }
    }

    // Now generate the output query.
    AbstractDataTypePtr output = c->getOutput();
    if (!isIntermediate(output)) {
        AbstractDataTypePtr queried = std::make_shared<const AbstractDataType>("_query_" + output->getName(),
                                                                               output->getType());
        new_ds[output] = queried;
        lowered.push_back(new QueryNode(output,
                                        queried,
                                        generateMetaDataFields(output, c)));
    }

    temp.push_back(new ComputeNode(c, new_ds));

    // Now generate all the intervals if any!
    std::vector<LowerIR> intervals = generateConsumesIntervals(c, temp);
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

void Pipeline::visit(const ComposeVec *c) {
    Pipeline inner_compose(c->compose);
    for (const auto &f : c->compose) {
        inner_compose.visit(f);
    }

    // If its a vec of vec...then, need to do something else.
    // right now, gern will complain if you try.
    // Now generate the outer loops.
    std::vector<LowerIR> intervals = generateOuterIntervals(
        to<FunctionCall>(c->compose[c->compose.size() - 1].ptr), inner_compose.getIRNodes());
    lowered.insert(lowered.end(), intervals.begin(), intervals.end());
}

void Pipeline::visit(const Pipeline *) {
}

bool Pipeline::isIntermediate(AbstractDataTypePtr d) const {
    (void)d;
    // This WILL change as I go beyond 1 function.
    return false;
}

std::vector<Expr> Pipeline::generateMetaDataFields(AbstractDataTypePtr d, FunctionCallPtr c) const {

    std::vector<Expr> metaFields;
    match(c->getAnnotation(), std::function<void(const SubsetNode *)>(
                                  [&](const SubsetNode *op) {
                                      if (op->data == d) {
                                          metaFields = op->mdFields;
                                      }
                                  }));
    if (!isIntermediate(d) || d == c->getOutput()) {
        return metaFields;
    } else {
        // Find the function that produces d
        // Solve the constraints for that function,
        // And return the solved constraints.
        throw error::InternalError("Unimplemented");
    }
}

std::vector<LowerIR> Pipeline::generateConsumesIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ConsumesForNode *)>(
                                  [&](const ConsumesForNode *op) {
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

std::vector<LowerIR> Pipeline::generateOuterIntervals(FunctionCallPtr f, std::vector<LowerIR> body) const {

    std::vector<LowerIR> current = body;
    match(f->getAnnotation(), std::function<void(const ComputesForNode *)>(
                                  [&](const ComputesForNode *op) {
                                      current = {new IntervalNode(op->start, op->end, op->step, current)};
                                  }));
    return current;
}

void Pipeline::accept(CompositionVisitor *v) const {
    v->visit(this);
}

void AllocateNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void FreeNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void InsertNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void QueryNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void ComputeNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void IntervalNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

void BlankNode::accept(PipelineVisitor *v) const {
    v->visit(this);
}

}  // namespace gern