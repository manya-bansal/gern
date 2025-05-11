cmake  -S . -B build -DOpenBLAS_DIR=/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas -DCMAKE_CUDA_HOST_COMPILER=g++
cmake --build build

