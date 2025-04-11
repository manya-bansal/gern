#pragma once

#include "compose/compose.h"

#include <string>

namespace gern::codegen::helpers {

/* 
The following code is used to check that the gern program has not violated
fundamental constraints on the device.
For more documentation: see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
*/

static std::string assert_device_constraints_decl = R"(

void assert_device_properties(int64_t grid_x_dim, 
                             int64_t grid_y_dim, 
                             int64_t grid_z_dim, 
                             int64_t block_x_dim, 
                             int64_t block_y_dim, 
                             int64_t block_z_dim,
                             int64_t smem_size) {
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    assert(device_properties.maxThreadsPerBlock >= block_x_dim * block_y_dim * block_z_dim);
    // assert(device_properties.warpSize == warp_dim);
    assert(device_properties.maxGridSize[0] >= grid_x_dim);
    assert(device_properties.maxGridSize[1] >= grid_y_dim);
    assert(device_properties.maxGridSize[2] >= grid_z_dim);
    assert(device_properties.maxThreadsDim[0] >= block_x_dim);
    assert(device_properties.maxThreadsDim[1] >= block_y_dim);
    assert(device_properties.maxThreadsDim[2] >= block_z_dim);
    assert(device_properties.sharedMemPerBlock >= smem_size);
}

)";

static FunctionCall assert_device_properties(Expr grid_x_dim,
                                             Expr grid_y_dim,
                                             Expr grid_z_dim,
                                             Expr block_x_dim,
                                             Expr block_y_dim,
                                             Expr block_z_dim,
                                             Expr smem_size) {
    return FunctionCall{
        .name = "assert_device_properties",
        .args = {
            Argument(grid_x_dim),
            Argument(grid_y_dim),
            Argument(grid_z_dim),
            Argument(block_x_dim),
            Argument(block_y_dim),
            Argument(block_z_dim),
            Argument(smem_size),
        },
        .template_args = {},
        .output = Parameter(),
        .grid = LaunchArguments(),
        .block = LaunchArguments(),
        .access = HOST,
        .smem_size = Expr(),
    };
}

}  // namespace gern::codegen::helpers