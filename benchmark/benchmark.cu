#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

/*
User includes the size of the square A and B in terminal line
*/

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
    // Destroy the handle
    cublasDestroy(handle);
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){ 
        for(int j = 0; j < nr_cols_A; ++j){
           std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]){

    if(argc != 2){
        std::cout << "The usage is ./bmark 'sizeof matrix' " << std::endl;
    }

    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = atoi(argv[1]);

    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    for(int i = 0; i<nr_rows_A; i++){
        for(int j = 0; j<nr_cols_A; j++){
            h_A[(j*nr_rows_A) + (i)] = (i*nr_rows_A) + (j+1);
            // std::cout << h_A[(i*nr_rows_A) + (j+1)] << " ";
            // if(j == nr_cols_A-1){
            //     std::cout << std::endl;
            // }
            h_B[(j*nr_rows_B) + (i)] = (i*nr_rows_A) + (j+1);
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);


    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 
    free(h_A);
    free(h_B);
    free(h_C);
 
    return 0;
}