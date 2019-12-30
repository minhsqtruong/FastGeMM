#include "fastgemm.cuh"
using namespace std;
int main(int argc, char const *argv[]) {

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  #ifdef DEBUG
  struct cudaDeviceProp prop;
  int device = 0;
  cudaGetDeviceProperties(&prop, device);
  cout << "Device name: " << prop.name << endl;
  cout << "Total Global Memory (bytes): " << prop.totalGlobalMem << endl;
  cout << "Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << endl;
  cout << "L1 Cache Size (bytes): 16000" << endl;
  cout << "L2 Cache Size (bytes): " << prop.l2CacheSize << endl;
  cout << "Cache Line Size (bytes): 128" << endl;
  cout << "Registers per Block: " << prop.regsPerBlock << endl;
  cout << "Warp Size: " << prop.warpSize << endl;
  cout << "Number of Warp Schedulers: 4" << endl;
  cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
  cout << "Compute Capability: " << prop.major << endl;
  cout << "Clock Rate (kHz): " << prop.clockRate << endl;
  cout << "Number of SM: " << prop.multiProcessorCount << endl;
  #endif

  // 1) Run reference code <DEBUG means run dumb CPU, REAL means CUBLAS>
  #ifdef DEBUG

  cout << "(M,N,K) = " << M << " " << N << " " << K << endl;

  float* C = (float*) malloc(sizeof(float) * M * N);
  float* A = (float*) malloc(sizeof(float) * M * K);
  float* B = (float*) malloc(sizeof(float) * K * N);

  for (int i = 0; i < M * N; i++)
    C[i] = 0.0;
  for (int i = 0; i < M * K; i++)
    A[i] = (float) i;
  for (int i = 0; i < M * N; i++)
    B[i] = (float) i;

  ref_mmul(C, A, B, M, N, K);

  cout << "Reference A: " << endl;
  printMatrix(A,M,K);
  cout << "Reference B: " << endl;
  printMatrix(B,K,N);
  cout << "Reference C: " << endl;
  printMatrix(C,M,N);

  // 2) Pack Data for Kernel <THIS CHANGE AS THE CODE PROGRESSES>
  cout << "Start initializing device arrays" << endl;
  float4* C_gpu;
  float4* A_gpu;
  float4* B_gpu;
  cout << cudaMallocManaged(&C_gpu, sizeof(float4) * (M * N)/4) << endl;;
  cudaMallocManaged(&A_gpu, sizeof(float4) * (M * K)/4);
  cudaMallocManaged(&B_gpu, sizeof(float4) * (K * N)/4);

  cout << "Start loading device arrays" << endl;
  for (int i = 0; i < (M * K)/4; i++)
    A_gpu[i] = make_float4(A[i*4 + 0],A[i*4 + 1],A[i*4 + 2],A[i*4 + 3]);
  for (int i = 0; i < (K * N)/4; i++)
    B_gpu[i] = make_float4(B[i*4 + 0],B[i*4 + 1],B[i*4 + 2],B[i*4 + 3]);

  cout << "Start fastgemm" << endl;
  launchFastGemm(C_gpu, A_gpu, B_gpu, M, N, K);

  free(A);
  free(B);
  free(C);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  #endif

  #ifdef REAL
  // INSERT CUBLAS HERE
  #endif

  return 0;
}

