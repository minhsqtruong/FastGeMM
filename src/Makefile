COMPILER=nvcc

all: kernel fastgemm

fastgemm: fastgemm.cu
	${COMPILER} -c fastgemm.cu -o fastgemm.o

main: main.cu fastgemm
	${COMPILER} -g -G fastgemm.o main.cu -o test

kernel: main
	./test 24 512 1

clean:
	rm -rf test
	rm -rf *.o

