#include "compose/runner.h"
#include "codegen/codegen.h"
#include "utils/error.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>

namespace gern {

void Runner::compile(Options config) {

    codegen::CodeGenerator cg(this->ordered_parameters);
    codegen::CGStmt code = cg.generate_code(c);
    signature = cg.getComputeFunctionSignature();

    config.prefix += "/";
    bool at_device = c.isDeviceLaunch();
    // std::string suffix = at_device ? ".cu" : ".cpp";
    std::string file = config.prefix + config.filename;
    std::ofstream outFile(file);
    outFile << code;
    outFile.close();

    std::string arch = at_device ? "-arch=sm_" + config.arch : "";
    std::string compiler = at_device ? "nvcc" : "g++";
    std::string compiler_option = at_device ? "--compiler-options " : "";
    std::string shared_obj = config.prefix + getUniqueName("libGern") + ".so";
    std::string cmd = compiler +
                      " -std=" + config.cpp_std + " " +
                      compiler_option +
                      " -fPIC " +
                      arch + " " + config.include +
                      " --shared -o " + shared_obj + " " +
                      file + " " + config.ldflags + " 2>&1";

    int runStatus = std::system(cmd.data());
    if (runStatus != 0) {
        throw error::UserError("Compilation Failed");
    }

    void *handle = dlopen(shared_obj.data(), RTLD_LAZY);
    if (!handle) {
        throw error::UserError("Error loading library: " + std::string(dlerror()));  // LCOV_EXCL_LINE
    }

    void *func = dlsym(handle, cg.getHookName().data());
    if (!func) {
        throw error::UserError("Error loading function: " + std::string(dlerror()));  // LCOV_EXCL_LINE
    }

    fp = (GernGenFuncPtr)func;
    argument_order = cg.getArgumentOrder();
    compiled = true;
}

void Runner::evaluate(std::map<std::string, void *> args) {
    if (!compiled) {
        throw error::UserError("Please compile the pipeline first");
    }

    size_t num_args = argument_order.size();
    if (args.size() != num_args) {
        throw error::UserError("Incorrect number of args passed! Expecting " + std::to_string(num_args) + " args");
    }
    // Now, fp has the FunctionSignature pointer,
    // and argument order contains the order
    // in which the arguments need to be set into
    // a void **.
    std::vector<void *> args_in_order;
    for (const auto &a : argument_order) {
        if (args.count(a) <= 0) {
            throw error::UserError("Argument " + a + "was not passed in");
        }
        args_in_order.push_back(args.at(a));
    }

    // Now, actually run the function.
    fp(args_in_order.data());
}

FunctionSignature Runner::getSignature() const {
    return signature;
}

void evaluate(Composable c, std::map<std::string, void *> args, Runner::Options config) {
    Runner runner(c);
    runner.compile(config);
    runner.evaluate(args);
}

}  // namespace gern