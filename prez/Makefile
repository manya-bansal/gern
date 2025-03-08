PROGRAM=1_trivial_use.cpu
BUILD_DIR=build

INCLUDE_DIR=/home/manya/gern/prez/library/array/impl

all: $(PROGRAM)

%.cpu: %.cpp $(BUILD_DIR).dir
	/usr/bin/c++ -std=c++20 -O3  \
	-I$(INCLUDE_DIR)  \
	-I ./ \
	-o $(BUILD_DIR)/$* $<

	./$(BUILD_DIR)/$*


%.dir:
	mkdir -p $* 




