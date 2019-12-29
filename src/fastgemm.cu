#include "fastgemm.cuh"

void printMatrix(float* mat, int row, int col)
{
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%6.1lf ", mat[i*col + j]);
    }
    printf("\n");
  }
}

void verify(float * C_ref, float * C, int  row, int col)
{
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (C_ref[i*col+j] != C[i*col+j]) {
        printf("ERROR at (%d,%d) \n", i, j);
        return;
      }
    }
  }
  printf("SUCCESS! no errors comparing with reference.\n");
}

void ref_mmul(float * C, float * A, float * B, int M, int N, int K)
{
    for (int k = 0; k < K; k++) {
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          C[m*N + n] += A[m*K + k] * B[k*N + n];
        }
      }
    }
}

__device__ __forceinline__ void outer_prod(float* C, float* shared_A, float* B, int id, int stride)
{
  // TODO Put C in register
  #pragma unroll
  for (int i = 0; i < M; i++)
    C[] += shared_A[] * B[];
}

__global__ void fastgemm(float4* C, float4* A, float4* B, int M, int N, int K)
{
  // INCORPERATE SHARED MEM IN L1 LOOP LATER
  extern __shared__ float4 shared_mem[];
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < M; i+=stride)
    shared_mem[i] = A[i];

  outer_prod(C, shared_mem, B, id, stride);
}

void launchFastGemm(float4* C, float4* A, float4* B, int M, int N, int K)
{
  int numBlocks  = 1;
  int numThreads = 128;
  int sharedSize = 49152; // Max size per block => M = 3072 floats
  fastgemm<<<numBlocks, numThreads, sharedSize>>>(C,A,B,M,N,K);
  cudaDeviceSynchronize();
}

