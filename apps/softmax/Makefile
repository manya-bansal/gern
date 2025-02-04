BUILD=build
FILENAME=main

all: bench

include Makefile.inc

INCLUDE += -I /home/manya/gern/test/ \
-I /home/manya/gern/test/library/matrix/impl/ \
-I /home/manya/gern/test/library/array/impl/  \
-I /home/manya/gpu-fusion/bench-cuda \
-I /home/manya/gern/apps/common


bench: $(FILENAME).run

