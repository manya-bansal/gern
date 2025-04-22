
BUILD = build

CUDA_DIR = /usr/local/cuda-12.3
CUBLAS_PATH = /usr/local/cuda-12.3/targets/x86_64-linux/lib/
CUDA_OBJDUMP = /usr/local/cuda-12.3/bin/cuobjdump
NVDIASM = /usr/local/cuda-12.3/bin/nvdisasm

all: smem_warp_tile.run

%.run: $(BUILD).dir
	nvcc --std=c++17 -O3 -g -o $(BUILD)/$* $*.cu -L$(CUBLAS_PATH) -lcublas
	./$(BUILD)/$*


clean:
	rm -f $(BUILD)

%.dir:
	mkdir -p $* 


%.sass: %.cu
	nvcc $(CUFLAGS) -Xptxas -v -arch=sm_89 -cubin $^ -o $(BUILD)/$*.cubin
	nvcc -ptx -arch=sm_89 $^ -o $(BUILD)/$*.ptx
	$(CUDA_OBJDUMP) -sass $(BUILD)/$*.cubin > $(BUILD)/$*.d
	$(NVDIASM) -hex $(BUILD)/$*.cubin > $(BUILD)/$*.elf

%.timeline: %.run
	sudo -E $(CUDA_DIR)/bin/nsys profile -t cuda --gpu-metrics-device=0 -o nsys-profile $(BUILD)/$*
	sudo -E /opt/nvidia/nsight-compute/2023.3.1/ncu --set full -f -o ncu-profile $(BUILD)/$*