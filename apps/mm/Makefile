BUILD=build
FILENAME=test
CUBLAS_DIR=/usr/local/cuda-12.6/targets/x86_64-linux/lib/

all: bench

include Makefile.inc

LDFLAGS +=-L $(CUBLAS_DIR)  -lcublas
INCLUDE +=-I /home/manya/gern/apps/common

bench: $(FILENAME).run

run: $(BUILD)
	echo $(BUILD)