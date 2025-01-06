#include "codegen/runner.h"
#include "codegen/codegen.h"
#include "utils/error.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>

namespace gern {

void Runner::compile(Options config) {

    p.lower();

    codegen::CodeGenerator cg;
    codegen::CGStmt code = cg.generate_code(p);

    config.prefix += "/";
    std::string file = config.prefix + config.filename;
    std::ofstream outFile(file);
    outFile << code;
    outFile.close();

    std::string shared_obj = config.prefix + getUniqueName("libGern") + ".so";
    std::string cmd = config.compiler +
                      " -std=c++11 --compiler-options -fPIC " +
                      config.includes +
                      " --shared -o " + shared_obj + " " +
                      file + " " + config.ldflags + " 2>&1";
    std::cout << cmd << std::endl;
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

void Runner::evaluate(std::map<std::string, void *> args) {
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

}  // namespace gern