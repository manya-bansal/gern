## GERN: Lightweight Fusion for GPUs

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

See the [CTest documentation] for more detail.

[CTest documentation]: https://cmake.org/cmake/help/latest/manual/ctest.1.html

[vcpkg]: https://vcpkg.io