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

// __global__ void fastgemm()
// {
//   kernel();
// }

void launchFastGemm()
{
  int numBlocks;
  int numThreads = 128;
  int sharedSize = 0;
  //fastgemm<<<numBlocks, numThreads, sharedSize>>>();
  cudaDeviceSynchronize();
}

