COMPILER=nvcc

all: main fastgemm

fastgemm: fastgemm.cu
	${COMPILER} -c fastgemm.cu -o fastgemm.o

main: main.cu fastgemm
	${COMPILER} -g -G fastgemm.o main.cu -o test

clean:
	rm -rf test
	rm -rf *.o

