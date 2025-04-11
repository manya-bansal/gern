#pragma once

#include "compose/compose.h"

namespace gern::codegen::helpers {

static std::string check_last_error_decl = R"(


void check_last_error() {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1); 
    }
}

)";

static FunctionCall check_last_error() {
    return FunctionCall{
        .name = "check_last_error",
        .args = {},
        .template_args = {},
        .output = Parameter(),
        .grid = LaunchArguments(),
        .block = LaunchArguments(),
        .access = HOST,
        .smem_size = Expr(),
    };
}

}  // namespace gern::codegen::helpers
