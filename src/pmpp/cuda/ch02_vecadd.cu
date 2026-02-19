/**
 * PMPP Chapter 2 - Vector Addition
 *
 * 本文件包含三个递进版本的向量加法:
 *   1. CPU 串行版本 (基线)
 *   2. CUDA 基础版本 (最简单的 GPU 实现)
 *   3. CUDA 完善版本 (含错误检查、计时、边界处理)
 *
 * 编译: nvcc -o ch02_vecadd ch02_vecadd.cu
 * 运行: ./ch02_vecadd [vector_size]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// 错误检查宏
// ============================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// 版本 1: CPU 串行向量加法 (基线)
// ============================================================

void vecadd_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// 版本 2: CUDA Kernel - 最基础版本
// ============================================================

__global__ void vecadd_kernel_basic(const float* A, const float* B,
                                     float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// 版本 3: CUDA Kernel - 使用 grid-stride loop
// 当向量很大时，一个线程可以处理多个元素，
// 减少 grid 大小，提高灵活性
// ============================================================

__global__ void vecadd_kernel_stride(const float* A, const float* B,
                                      float* C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// Host 端完整流程: 内存分配 → 数据传输 → Kernel → 取回结果
// ============================================================

void vecadd_gpu(const float* h_A, const float* h_B, float* h_C, int n) {
    size_t size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Step 1: 分配 Device 内存
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Step 2: Host → Device 数据传输
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Step 3: 配置并启动 Kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vecadd_kernel_basic<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Device → Host 数据传输
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Step 5: 释放 Device 内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================
// 验证与计时
// ============================================================

void init_random(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

int verify(const float* ref, const float* result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - result[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: CPU=%.6f, GPU=%.6f\n",
                    i, ref[i], result[i]);
            return 0;
        }
    }
    return 1;
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1 << 24;  // 默认 16M 元素

    printf("=== PMPP Ch02: Vector Addition ===\n");
    printf("Vector size: %d (%.1f MB per vector)\n\n",
           n, n * sizeof(float) / (1024.0 * 1024.0));

    // 打印 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("  SMs: %d, Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("  Global memory: %.1f GB\n\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // 分配 Host 内存
    size_t size = n * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);

    srand(42);
    init_random(h_A, n);
    init_random(h_B, n);

    // CPU 基线
    double t0 = get_time_ms();
    vecadd_cpu(h_A, h_B, h_C_cpu, n);
    double t1 = get_time_ms();
    printf("[CPU] Time: %.2f ms\n", t1 - t0);

    // GPU 版本
    double t2 = get_time_ms();
    vecadd_gpu(h_A, h_B, h_C_gpu, n);
    double t3 = get_time_ms();
    printf("[GPU] Time: %.2f ms (including H2D + D2H transfers)\n", t3 - t2);

    // 验证
    if (verify(h_C_cpu, h_C_gpu, n)) {
        printf("\n[PASS] GPU result matches CPU result.\n");
    } else {
        printf("\n[FAIL] Results mismatch!\n");
    }

    // GPU kernel-only 计时 (不含传输)
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 预热
    vecadd_kernel_basic<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    vecadd_kernel_basic<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("[GPU Kernel-only] Time: %.3f ms\n", kernel_ms);

    double bandwidth_gb = 3.0 * n * sizeof(float) / (kernel_ms / 1000.0) / 1e9;
    printf("[GPU Kernel-only] Effective bandwidth: %.1f GB/s\n", bandwidth_gb);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
