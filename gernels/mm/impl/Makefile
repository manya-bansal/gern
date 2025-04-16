
BUILD = build

all: $(BUILD).dir
	nvcc --std=c++17 -O3 -g -o $(BUILD)/smem_new smem_new.cu 
	./$(BUILD)/smem_new


clean:
	rm -f $(BUILD)

%.dir:
	mkdir -p $* 
