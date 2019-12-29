#include "fastgemm.cuh"
using namespace std;
int main(int argc, char const *argv[]) {

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
  return 0;
}

