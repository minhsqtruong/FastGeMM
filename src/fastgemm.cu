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

// Have to look into if this is neccessary
__device__ __forceinline__ int conflict_free_index(int local_id, int real_idx)
{
  return real_idx * NUM_THREADS + local_id;
}


__device__ __forceinline__ void outer_prod(float* C, float* A, float4* B, int id, int stride)
{
  float4 b = B[id];

  #pragma unroll
  for (int m = 0; m < M; m++) {
    C[conflict_free_index(id, m*4 + 0)] = fmaf(A[m], b.x, C[conflict_free_index(id, m*4 + 0)]);
    C[conflict_free_index(id, m*4 + 1)] = fmaf(A[m], b.y, C[conflict_free_index(id, m*4 + 1)]);
    C[conflict_free_index(id, m*4 + 2)] = fmaf(A[m], b.z, C[conflict_free_index(id, m*4 + 2)]);
    C[conflict_free_index(id, m*4 + 3)] = fmaf(A[m], b.w, C[conflict_free_index(id, m*4 + 3)]);
  }
}

__global__ void fastgemm(float* C, float4* A, float4* B)
{ // Assuming K = 1 for now

  // Memory Instantiation
  extern __shared__ float sharedMem[];
  float registers_0[M];
  float registers_1[M];
  float* this_registers;
  float* next_registers;
  float* tmp;
  float4* A_vec;
  float4* B_vec;

  // Identification
  int id = threadIdx.x;
  int stride = blockDim.x;

  // Load C (Incorperate into L1 later)
  for (int i = id; i < MAX_SHARED_SIZE_FLOAT; i+=stride)
    sharedMem[i] = 0.0;

  // Preload Setup
  for (int m = 0; m < MBY4; m+= 1) {
    float4 num = A[m];
    registers_0[m*4 + 0] = num.x;
    registers_0[m*4 + 1] = num.y;
    registers_0[m*4 + 2] = num.z;
    registers_0[m*4 + 3] = num.w;
  }
  next_registers = registers_0;
  this_registers = registers_1;

  for (int k = 1; k < K; k++) {
    // Ping pong for preload
    tmp = this_registers;
    this_registers = next_registers;
    next_registers = tmp;
    A_vec = A + k*MBY4;

    // Preload the next set of A_vec in
    #pragma unroll
    for (int m = 0; m < MBY4; m+= 1) {
      float4 num = A_vec[m];
      next_registers[m*4 + 0] = num.x;
      next_registers[m*4 + 1] = num.y;
      next_registers[m*4 + 2] = num.z;
      next_registers[m*4 + 3] = num.w;
    }
    B_vec = B + (k-1)*NBY4;
    outer_prod(sharedMem, this_registers, B_vec, id, stride);
  }

  // Need to turn C into float4 later
  if (id == 0)
    printf("Store C back\n");
  for (int i = id; i < MAX_SHARED_SIZE_FLOAT; i+=stride)
    C[i] = sharedMem[i];
}

void launchFastGemm(float* C, float4* A, float4* B)
{
  fastgemm<<<NUM_BLOCKS, NUM_THREADS, MAX_SHARED_SIZE_BYTES>>>(C,A,B);
  cudaDeviceSynchronize();
}

