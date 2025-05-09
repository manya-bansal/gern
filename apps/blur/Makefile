# Makefile for Halide build.
HALIDE_PATH=/home/manya/halide_local
HALIDE_BUILD=build_halide

CXX=g++-10
CXX_FLAGS=-std=c++17 -O3

PROGRAM=halide_blur


all: $(HALIDE_BUILD).dir halide_gen halide_run 
	$(HALIDE_BUILD)/halide_run

halide_gen: $(PROGRAM)_generator.cpp
	$(CXX) $(CXX_FLAGS) $< \
	    -I $(HALIDE_PATH)/include \
	    -I $(HALIDE_PATH)/share/tools \
		-Wl,-rpath=$(HALIDE_PATH)/lib \
		-L $(HALIDE_PATH)/lib -lHalide \
		-rdynamic \
	   	-Wl,--whole-archive -lHalide_GenGen -Wl,--no-whole-archive \
		-o $(HALIDE_BUILD)/$@ -ldl
	
		./$(HALIDE_BUILD)/$@ -g $(PROGRAM) \
		-e static_library,c_header,c_source,registration,stmt \
		-o ./$(HALIDE_BUILD) target=host-cuda

halide_run: $(PROGRAM)_main.cpp 
	$(CXX) $(CXX_FLAGS) $< \
		./$(HALIDE_BUILD)/$(PROGRAM).a \
		-g -std=c++17 -O3 \
		-I ./$(HALIDE_BUILD) \
		-I /usr/local/cuda-12.3/include \
		-I $(HALIDE_PATH)/include \
		-I $(HALIDE_PATH)/share/tools \
		-L $(HALIDE_PATH)/lib \
		-Wl,-rpath=$(HALIDE_PATH)/lib -lHalide \
		`libpng-config --cflags --ldflags` -ljpeg  -o $(HALIDE_BUILD)/$@ \
		-Wl,-rpath,/usr/local/cuda-12.3/lib64 -L/usr/local/cuda-12.3/lib64 \
		-ldl -lpthread -lcudart

gern: 
	nvcc -O3 -arch=sm_89 \
	-I  /home/manya/gern/apps/common \
	-I  /home/manya/gern/test/ \
	blur.cu  -o $(HALIDE_BUILD)/gern_blur



%.dir:
	mkdir -p $*


.PHONY: clean
clean:
	rm -rf $(HALIDE_BUILD)
