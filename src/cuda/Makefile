CXX=g++
NVCC=nvcc
CFLAGS=-g -O0 -Wall -std=c++17
NVCFLAGS=-g -O0 --compiler-options -Wall --std=c++14
LD_FLAGS=-lcuda -lcudart -lcublas

OBJECTS=cbsconv.o tensor.o timer.o

DEPENDENCY_RULES=$(OBJECTS:=.d) $(MAIN_OBJECTS:=.d)

all: main

main: main.o $(OBJECTS)
	$(NVCC) $(LD_FLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CFLAGS) -c $< -o $@
	$(CXX) $(CFLAGS) -M $< > $@.d

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c $< -o $@
	$(NVCC) $(NVCFLAGS) -M $< > $@.d

-include $(DEPENDENCY_RULES)

clean:
	rm -rf *.o *.d cbsconvdd

.PHONY: all FORCE
