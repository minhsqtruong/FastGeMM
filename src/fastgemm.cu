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

void ref_mmul(float * C, float * A, float * B)
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
  return real_idx * NUM_THREADS + local_id;
}


__device__ __forceinline__ void outer_prod(float* C, float* A, float4* B, int id, int stride)
{
  float4 b = B[id];

  #pragma unroll
  for (int m = 0; m < M; m++) {
    fmaf(C[conflict_free_index(id, m*4 + 0)],A[m], b.x);
    fmaf(C[conflict_free_index(id, m*4 + 1)],A[m], b.y);
    fmaf(C[conflict_free_index(id, m*4 + 2)],A[m], b.z);
    fmaf(C[conflict_free_index(id, m*4 + 3)],A[m], b.w);
  }
}

__global__ void fastgemm(float4* C, float4* A, float4* B)
{ // Assuming K = 1 for now

  // Memory Instantiation
  extern __shared__ float sharedMem[];
  float registers[M];
  float4* A_vec;
  float4* B_vec;

  // Identification
  int id = threadIdx.x;
  int stride = blockDim.x;

  // Load C (Incorperate into L1 later)
  for (int i = id; i < MAX_SHARED_SIZE_FLOAT4; i+=stride)
    sharedMem[i] = 0.0;

  for (int k = 0; k < K; k++) {
    // Load A (Use Preload technique later)
    #pragma unroll
    for (int m = 0; m < MBY4; m+= 1) {
      A_vec = A + k*MBY4;
      B_vec = B + k*NBY4;
      float4 num = A_vec[m];
      registers[m*4 + 0] = num.x;
      registers[m*4 + 1] = num.y;
      registers[m*4 + 2] = num.z;
      registers[m*4 + 3] = num.w;
    }
    outer_prod(sharedMem, registers, B_vec, id, stride);
  }
}

void launchFastGemm(float4* C, float4* A, float4* B)
{
  fastgemm<<<NUM_BLOCKS, NUM_THREADS, MAX_SHARED_SIZE_BYTES>>>(C,A,B);
  cudaDeviceSynchronize();
}

