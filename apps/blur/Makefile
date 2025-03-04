# Makefile for Halide build.

BUILD=build
BIN=bin
HALIDE_PATH=/home/manya/halide_local



CXX=clang++-17
GLIBC=/snap/core18/2855/lib/x86_64-linux-gnu/


halide: halide_blur_generator.cpp $(BUILD).dir $(BIN).dir
	$(CXX) -std=c++17 -O3 $< \
	    -I $(HALIDE_PATH)/include \
	    -I $(HALIDE_PATH)/share/tools \
		-Wl,-rpath=$(HALIDE_PATH)/lib \
		-L $(HALIDE_PATH)/lib -lHalide \
	   	-Wl,--whole-archive -lHalide_GenGen -Wl,--no-whole-archive \
		-o $(BIN)/$@
	
		./$(BIN)/$@ -g halide_blur \
		-e static_library,c_header,c_source,registration,stmt \
		-r gpu \
		-o ./$(HALIDE_BUILD) target=host

%.dir:
	mkdir -p $*
