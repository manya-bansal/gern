
BUILD = build

all: $(BUILD).dir
	nvcc -g -o $(BUILD)/test test.cu 
	./$(BUILD)/test


clean:
	rm -f $(BUILD)

%.dir:
	mkdir -p $* 
