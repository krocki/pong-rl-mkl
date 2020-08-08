GCC=gcc

# use local OpenBLAS
#OPENBLAS=./OpenBLAS

#OPENBLAS
#C_FLAGS=-O3 -fPIC -I${OPENBLAS}
#L_FLAGS=${OPENBLAS}/libopenblas.a

#MKL
MKL_PATH=/opt/intel
C_FLAGS=-O3 -fPIC -I${MKL_PATH}/mkl/include
L_FLAGS=-L${MKL_PATH}/lib -L${MKL_PATH}/mkl/lib -liomp5 -lmkl_rt

all: nn.so

nn.so: nn.o nn.c Makefile
	${GCC} nn.o ${L_FLAGS} -shared -o $@

%.o: %.c
	${GCC} ${C_FLAGS} -c $< -o $@

clean:
	rm -rf *.o *.so
