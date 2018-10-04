CUDA_INSTALL_PATH = /opt/cuda
CC = gcc
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = game_of_life_cuda game_of_life_cuda.o 

game_of_life_cuda: game_of_life_cuda.o cuda_kernel_func.o
	${CC} ${LFLAGS} -o $@ $^
	cp $@ ../release

game_of_life_cuda.o: game_of_life_cuda.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@

cuda_kernel_func.o: cuda_kernel_func.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@


clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/simple_add

rebuild: clean game_of_life_cuda

