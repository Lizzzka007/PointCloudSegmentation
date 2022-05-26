GDB ?= no
FF ?= yes #FF == Fast flags

FLAGS = 
NVCCFLAGS = 

ifeq ($(GDB), yes)
	FLAGS += -g
	NVCCFLAGS += -g
endif

ifneq ($(FF), no)
	FLAGS += -std=c++2a -O3 -ffast-math -ffinite-math-only -Wunused-but-set-variable
endif

CC_LINK_LIB = -L/usr/local/cuda/lib64 
CC_LINK_FLAGS = -lcuda -lcudart -lcublas

Geo = ../Src/Geometry
Src = ../Src

a.out: VectorLib.o TripleClosestPoints.cuobj main.o
	g++ $(FLAGS) VectorLib.o TripleClosestPoints.cuobj main.o -o a.out $(CC_LINK_LIB) -lcuda -lcudart -lcublas

VectorLib.o: $(Geo)/VectorLib.cpp $(Geo)/Tune.h $(Geo)/Similart.h $(Geo)/Vector2.h $(Geo)/Vector.h
	g++ -c $(FLAGS) $(Geo)/VectorLib.cpp -o VectorLib.o

TripleClosestPoints.cuobj: ../CUDAsrc/TripleClosestPoints/TripleClosestPoints.cu ../CUDAsrc/TripleClosestPoints/TripleClosestPoints.cuh $(Geo)/Vector.h
	nvcc -arch=sm_60 -c $(NVCCFLAGS) ../CUDAsrc/TripleClosestPoints/TripleClosestPoints.cu -o TripleClosestPoints.cuobj -lcublas

main.o: main.cpp ../CUDAsrc/TripleClosestPoints/TripleClosestPoints.cuh $(Geo)/Vector.h
	g++ -c $(FLAGS) main.cpp -o main.o

clean :
	-rm -f *.o  
	-rm -f *.cuobj  
	-rm -f *.txt
	-rm -f *.out
