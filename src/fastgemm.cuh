#include <iostream>
#include <stdio.h>

/*============*/
#define DEBUG
/*==============*/

/*CPU RELATED FUNCTIONS*/
void ref_mmul(float*, float*, float*, int, int ,int);
void verify(float*, float*, int, int);
void printMatrix(float*, int, int);

/*GPU RELATED FUNCTIONS*/
__device__ __forceinline__ void kernel();
__device__ __forceinline__ void L1Loop();
__device__ __forceinline__ void L2Loop();

__global__ void fastgemm();
void launchFastGemm();

