# Makefile for Halide build.

BUILD=build
BIN=bin
HALIDE_PATH=/home/manya/Halide-19.0.0-x86-64-linux



CXX=clang++-17
GLIBC=/snap/core18/2855/lib/x86_64-linux-gnu/


halide: halide_blur_generator.cpp $(BUILD).dir $(BIN).dir
	$(CXX) -std=c++17 -O3 \
	    -I $(HALIDE_PATH)/include \
	    -I $(HALIDE_PATH)/share/Halide/tools \
		-Wl,--dynamic-linker="${GLIBC}/lib/ld-linux-x86-64.so.2" \
	    -Wl,-rpath=$(GLIBC) -L$(GLIBC) -lc -ldl \
		-L  /usr/lib/x86_64-linux-gnu/ -lc_nonshared \
	    -L $(HALIDE_PATH)/lib -lHalide \
	   	-Wl,--whole-archive -lHalide_GenGen -Wl,--no-whole-archive \
		-o $(BIN)/$@
	


	# ./$(BIN)/$@ -g blur_generator \
	# -e static_library,c_header,registration,stmt \
	# -o ./$(HALIDE_BUILD) target=host

%.dir:
	mkdir -p $*
