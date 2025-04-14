#!/bin/bash

# Run the benchmark for the given kernel.
# Usage: ./bench.sh <kernel>

# Check if the kernel is provided.
if [ -z "$1" ]; then
    echo "Usage: ./bench.sh <kernel>"
    exit 1
fi

range_1=(128 512 1024 2048 4096)

for m in ${range_1[@]}; do
    # First let's write the value to a .h file.
    echo "Generating kernel_${1}.h for ${m}x${m}x${m}..."

    echo "#define M_CONST ${m}" > "kernel_${1}.h"
    echo "#define N_CONST ${m}" >> "kernel_${1}.h"
    echo "#define K_CONST ${m}" >> "kernel_${1}.h"

    # # Now, let's compile the kernel.
    (cd ../ && cd build && make)

    # ## Finally, let's run
    (cd ../ && cd build && ./mm/kernel_${1})
done
