#include "fastgemm.cuh"

void printMatrix(float* mat, int row, int col)
{
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (j < 10)
        printf("%6.1lf ", mat[i*col + j]);
      else {
        printf(" ...");
        break;
      }
    }
    printf("\n");
    if(i > 10)
      break;
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

__device__ __forceinline__ int conflict_free_index(int local_id, int real_idx)
{
  return real_idx * 128 + local_id;
}


__device__ __forceinline__ void outer_prod(float* C, float* A, float4* B, int id, int stride)
{
  float4 b = B[id];

  #pragma unroll
  for (int i = 0; i < 24; i++) {
    fmaf(C[conflict_free_index(id, i*4 + 0)],A[i], b.x);
    fmaf(C[conflict_free_index(id, i*4 + 1)],A[i], b.y);
    fmaf(C[conflict_free_index(id, i*4 + 2)],A[i], b.z);
    fmaf(C[conflict_free_index(id, i*4 + 3)],A[i], b.w);
  }
}

__global__ void fastgemm(float4* C, float4* A, float4* B)
{ // Assuming K = 1 for now

  // Memory Instantiation
  extern __shared__ float sharedMem[];
  float registers[24];

  // Identification
  int id = threadIdx.x;
  int stride = blockDim.x;

  // Load (Incorperate into L1 later)
  for (int i = id; i < 12288; i+=stride)
    sharedMem[i] = 0.0;
  #pragma unroll
  for (int i = 0; i < 6; i+= 1) {
    float4 num = A[i];
    registers[i*4 + 0] = num.x;
    registers[i*4 + 1] = num.y;
    registers[i*4 + 2] = num.z;
    registers[i*4 + 3] = num.w;
  }

  outer_prod(sharedMem, registers, B, id, stride);
}

void launchFastGemm(float4* C, float4* A, float4* B, int M, int N, int K)
{
  int numBlocks  = 1;
  int numThreads = 128;
  int sharedSize = 49152; // Max size per block
  fastgemm<<<numBlocks, numThreads, sharedSize>>>(C,A,B);
  cudaDeviceSynchronize();
}

