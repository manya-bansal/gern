
BUILD = build

all: smem_warp_tile.run

%.run: $(BUILD).dir
	nvcc --std=c++17 -O3 -g -o $(BUILD)/$* $*.cu
	./$(BUILD)/$*


clean:
	rm -f $(BUILD)

%.dir:
	mkdir -p $* 
