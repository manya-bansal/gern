## GERN: Lightweight Fusion for GPUs

🚧 **Under Construction!** 🚧

To build gern, install [vcpkg] and make sure `VCPKG_ROOT` is set in your
environment. Then run:

```shell
$ cmake -DGern_CUDA_ARCH=<89,90..,etc> --preset dev
$ cmake --build build/dev
```

If  ```-DGern_CUDA_ARCH``` is not set, none
of the GPU kernels will be run during tests.

To run tests:

```shell
$ ctest --test-dir build/dev
```

Or to run a single test:

```shell
$ ctest --test-dir build/dev -R ExprNode.Literal
```

### Testing Code Coverage

To test with code coverage, build with preset `coverage`:

```
$ cmake -DGern_CUDA_ARCH=<89,90..,etc> --preset coverage 
$ cmake --build build/coverage
$ ctest --test-dir build/coverage
```

Then, generate an html for the code coverage results from `build/coverage`:

```
$ gcovr -r  ~/gern/src/ CMakeFiles/Gern_Gern.dir/src/ --exclude-unreachable-branches --html-details -o <location>
```

See the [CTest documentation] for more detail.

[CTest documentation]: https://cmake.org/cmake/help/latest/manual/ctest.1.html

[vcpkg]: https://vcpkg.io
