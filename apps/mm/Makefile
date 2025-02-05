BUILD=build
FILENAME=sgemm
CUBLAS_DIR=/usr/local/cuda-12.6/targets/x86_64-linux/lib/

all: bench

include Makefile.inc

LDFLAGS +=-L $(CUBLAS_DIR)  -lcublas

bench: $(FILENAME).bin

run: $(BUILD)
	echo $(BUILD)