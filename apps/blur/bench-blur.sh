impls=(
    "gern"
)

> gern
> halide

for impl in ${impls[@]}; do
    echo "Benchmarking $impl"
    for i in $(seq 2 2 40); do
        echo "#define size 128 * $i" > value.h
        make -B gern
        echo  $((128 * i)) >> gern
        ./halide_build/gern_blur >> gern
        make halide_gen
        make halide_run
        ./halide_build/halide_run >> halide
    done
done