
BUILD = build

all: smem_2d.run

%.run: $(BUILD).dir
	nvcc --std=c++17 -O3 -g -o $(BUILD)/$* $*.cu
	./$(BUILD)/$*


clean:
	rm -f $(BUILD)

%.dir:
	mkdir -p $* 
