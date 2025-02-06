# Need to define where the cuda compiler is located
CUDA_DIR = /usr/local/cuda-12.6/


CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc
CUDA_OBJDUMP = $(CUDA_DIR)/bin/cuobjdump
NVDIASM = $(CUDA_DIR)/bin/nvdisasm
CUFLAGS = -O3 --default-stream per-thread -arch sm_89 -std=c++17
INCLUDE+= 
LDFLAGS+=

$(BUILD):
	mkdir $(BUILD)

$(DATA):
	mkdir $(DATA)

%.bin: %.cu $(BUILD)
	$(CUDA_COMPILER) $(CUFLAGS)  $*.cu $(INCLUDE)  \
	$(LDFLAGS) -o $(BUILD)/$*

%.run: %.bin
	$(BUILD)/$*

%.sass: %.cu
	$(CUDA_COMPILER) $(CUFLAGS) $^  $(INCLUDE) $(LDFLAGS) \
	-Xptxas -v -arch=sm_89 -cubin -o $(BUILD)/$*.cubin
	$(CUDA_COMPILER) $(CUFLAGS) $^  $(INCLUDE) $(LDFLAGS) \
	-Xptxas -v -arch=sm_89 -ptx -o $(BUILD)/$*.ptx
	$(CUDA_OBJDUMP) -sass $(BUILD)/$*.cubin > $(BUILD)/$*.d
	$(NVDIASM) -hex $(BUILD)/$*.cubin > $(BUILD)/$*.elf

%.profile: %.cu %.bin
	sudo -E /opt/nvidia/nsight-compute/2023.3.1//ncu \
	--set full -f -o profile $(BUILD)/$* gern

clean: 
	rm -rf $(BUILD)

clean-profile:
	rm -rf *.*rep
