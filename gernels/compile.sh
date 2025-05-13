cmake  -S . -B build -DMKL_DIR=/opt/intel/oneapi/mkl/2025.1/lib/cmake/mkl -DCMAKE_CUDA_HOST_COMPILER=g++ 
cmake --build build

