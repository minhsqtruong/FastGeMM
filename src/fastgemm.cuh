#include <iostream>
#include <stdio.h>
#include <stdlib.h>

/*============*/
#define DEBUG
//#define REAL
/*==============*/

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
void launchFastGemm(float4*, float4*, float4*, int, int, int);

