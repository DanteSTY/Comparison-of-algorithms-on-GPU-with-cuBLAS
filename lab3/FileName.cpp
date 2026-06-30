#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono> 
#include <iostream>
#include <cmath>


#define N 1024          
#define TILE_SIZE 32    


void matrixMulCPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


__global__ void matrixMulNaive(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}


__global__ void matrixMulShared(float* A, float* B, float* C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;


    for (int m = 0; m < (n / TILE_SIZE); ++m) {
        
        As[ty][tx] = A[row * n + (m * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(m * TILE_SIZE + ty) * n + col];

        __syncthreads(); 

        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); 
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}


void checkResult(float* hostRef, float* gpuRef, int n) {
    double epsilon = 1.0E-4;
    for (int i = 0; i < n * n; i++) {
        if (std::abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Mismatch at index %d: CPU %f != GPU %f\n", i, hostRef[i], gpuRef[i]);
            return;
        }
    }
    printf("PASSED (Results match CPU)\n");
}

int main() {
    int size = N * N * sizeof(float);
    float* h_A, * h_B, * h_C_CPU, * h_C_Naive, * h_C_Shared, * h_C_cuBLAS;

    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_CPU = (float*)malloc(size);
    h_C_Naive = (float*)malloc(size);
    h_C_Shared = (float*)malloc(size);
    h_C_cuBLAS = (float*)malloc(size);

    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    printf(">>> Matrix Multiplication Comparison (%dx%d) <<<\n", N, N);
    printf("--------------------------------------------------\n");

    
    printf("1. Running CPU implementation... ");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    printf("Done!\n");
    printf("   CPU Time:       %f ms\n\n", cpu_duration.count());

    
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;


    
    cudaEventRecord(start);
    matrixMulNaive << <blocks, threads >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C_Naive, d_C, size, cudaMemcpyDeviceToHost);

    printf("2. GPU Naive:      %f ms | ", milliseconds);
    checkResult(h_C_CPU, h_C_Naive, N);

    
    cudaEventRecord(start);
    matrixMulShared << <blocks, threads >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C_Shared, d_C, size, cudaMemcpyDeviceToHost);

    float shared_time = milliseconds; 
    printf("3. GPU Shared:     %f ms | ", shared_time);
    checkResult(h_C_CPU, h_C_Shared, N);

    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);

    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("4. cuBLAS:         %f ms (Warm)\n", milliseconds);
    cublasDestroy(handle);

    printf("--------------------------------------------------\n");
    printf("SPEEDUP (GPU Shared vs CPU):  %.2fx FASTER\n", cpu_duration.count() / shared_time);
    printf("--------------------------------------------------\n");

    
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_Naive); free(h_C_Shared); free(h_C_cuBLAS);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
