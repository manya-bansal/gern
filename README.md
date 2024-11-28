## GERN: Lightweight Fusion for GPUs

To build gern, run: 

```
git submodule update --init --recursive
cd symengine
mkdir build 
cd build && cmake .. & make
cd ../../
mkdir build && cd build
cmake .. 
make -j 8
```

To run test, run in the ```build/test``` directory:

```
./gern_test
```

Since Gern uses google test as its testing infrastructure, it is also
possible to passing in a regular expression. For eg. 
```./gern_test --gtest_filer=<RegEx>```.
