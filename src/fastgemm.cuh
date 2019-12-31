#include <iostream>
#include <stdio.h>
#include <stdlib.h>

/*============*/
#define DEBUG
//#define REAL
/*==============*/

#define M 24
#define N 512
#define K 1
#define MBY4 6
#define NBY4 128
#define MAX_SHARED_SIZE_BYTES 49152
#define MAX_SHARED_SIZE_FLOAT4 12288
#define NUM_THREADS 128
#define NUM_BLOCKS 1

/*CPU RELATED FUNCTIONS*/
void ref_mmul(float*, float*, float*, int, int ,int);
void verify(float*, float*, int, int);
void printMatrix(float*, int, int);

/*GPU RELATED FUNCTIONS*/
__device__ __forceinline__ int conflict_free_index(int, int);
__device__ __forceinline__ void outer_prod(float*, float*, float4*, int, int);
__device__ __forceinline__ void L1Loop();
__device__ __forceinline__ void L2Loop();

__global__ void fastgemm(float4*, float4*, float4*);
void launchFastGemm(float4*, float4*, float4*);

