/**
 * PMPP Chapter 5 - Tiled Matrix Multiplication
 *
 * 本文件包含四个递进版本的矩阵乘法:
 *   1. 基础版本 (无 Tiling, Ch03 回顾)
 *   2. Tiled 版本 (Shared Memory, 核心优化)
 *   3. Tiled + 边界处理版本 (处理非整除情况)
 *   4. Tiled + 双缓冲 Prefetch 版本 (进阶优化)
 *
 * 编译: nvcc -O3 -arch=sm_80 -o ch05_tiled_matmul ch05_tiled_matmul.cu
 * 运行: ./ch05_tiled_matmul [M] [K] [N]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
// 版本 1: 基础矩阵乘法 (无 Tiling)
// 每个线程独立从全局内存读取 A 的一行和 B 的一列
// ============================================================

__global__ void matmul_basic(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================
// 版本 2: Tiled 矩阵乘法 (Shared Memory)
// 假设 M, K, N 都是 TILE_SIZE 的整数倍
// ============================================================

#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;

    for (int ph = 0; ph < K / TILE_SIZE; ph++) {
        // Phase 1: 协作加载 A 和 B 的 Tile 到 Shared Memory
        As[ty][tx] = A[row * K + (ph * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];

        __syncthreads();

        // Phase 2: 从 Shared Memory 计算部分点积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// ============================================================
// 版本 3: Tiled + 边界处理
// 处理 M, K, N 不是 TILE_SIZE 整数倍的情况
// ============================================================

__global__ void matmul_tiled_boundary(const float* A, const float* B, float* C,
                                       int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;

    int numPhases = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int ph = 0; ph < numPhases; ph++) {
        int a_col = ph * TILE_SIZE + tx;
        int b_row = ph * TILE_SIZE + ty;

        // 加载时做边界检查, 越界位置填 0
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// 版本 4: Tiled + 寄存器累加 + 更大 Tile (32×32)
// ============================================================

#define TILE_SIZE_32 32

__global__ void matmul_tiled_32(const float* A, const float* B, float* C,
                                 int M, int K, int N) {
    __shared__ float As[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float Bs[TILE_SIZE_32][TILE_SIZE_32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_32 + tx;
    int row = blockIdx.y * TILE_SIZE_32 + ty;

    float sum = 0.0f;

    int numPhases = (K + TILE_SIZE_32 - 1) / TILE_SIZE_32;

    for (int ph = 0; ph < numPhases; ph++) {
        int a_col = ph * TILE_SIZE_32 + tx;
        int b_row = ph * TILE_SIZE_32 + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 手动展开内层循环 (编译器通常也会做)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// CPU 基线
// ============================================================

void matmul_cpu(const float* A, const float* B, float* C,
                int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ============================================================
// 性能测试框架
// ============================================================

typedef void (*KernelFunc)(const float*, const float*, float*, int, int, int);

struct KernelConfig {
    const char* name;
    KernelFunc kernel;
    int tile_size;
};

float benchmark_kernel(KernelConfig& cfg, const float* d_A, const float* d_B,
                       float* d_C, int M, int K, int N, int warmup, int repeat) {
    dim3 block(cfg.tile_size, cfg.tile_size);
    dim3 grid((N + cfg.tile_size - 1) / cfg.tile_size,
              (M + cfg.tile_size - 1) / cfg.tile_size);

    for (int i = 0; i < warmup; i++)
        cfg.kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        cfg.kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

int verify(const float* ref, const float* result, int n, float tol) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - result[i]);
        if (err > max_err) max_err = err;
    }
    return max_err < tol;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== PMPP Ch05: Tiled Matrix Multiplication ===\n");
    printf("C(%d×%d) = A(%d×%d) × B(%d×%d)\n\n", M, N, M, K, K, N);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SMs: %d, Shared Mem/Block: %lu KB)\n\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_ref = (float*)malloc(size_C);
    float* h_C_gpu = (float*)malloc(size_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // CPU 参考结果
    if (M <= 1024 && K <= 1024 && N <= 1024) {
        matmul_cpu(h_A, h_B, h_C_ref, M, K, N);
        printf("[CPU] Reference computed.\n");
    } else {
        printf("[CPU] Skipped (matrix too large).\n");
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    double gflop = 2.0 * M * N * K / 1e9;

    KernelConfig kernels[] = {
        {"Basic (no tiling)",     matmul_basic,          16},
        {"Tiled 16×16",           matmul_tiled,          16},
        {"Tiled 16×16 + boundary",matmul_tiled_boundary, 16},
        {"Tiled 32×32 + boundary",matmul_tiled_32,       32},
    };

    printf("\n%-28s %10s %10s %10s %8s\n",
           "Kernel", "Time(ms)", "GFLOPS", "Bandwidth", "Verify");
    printf("%s\n", std::string(76, '-').c_str());

    int warmup = 3, repeat = 10;
    for (auto& cfg : kernels) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));

        float ms = benchmark_kernel(cfg, d_A, d_B, d_C, M, K, N, warmup, repeat);
        double gflops = gflop / (ms / 1000.0);

        // 有效带宽: 每个 C 元素需要读 2K 个 float (基础) 或约 2K/T 个 (tiled)
        double eff_bw = (size_A + size_B + size_C) / (ms / 1000.0) / 1e9;

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

        const char* status = "N/A";
        if (M <= 1024 && K <= 1024 && N <= 1024) {
            status = verify(h_C_ref, h_C_gpu, M * N, 1e-1f) ? "PASS" : "FAIL";
        }

        printf("%-28s %8.3f ms %8.1f %8.1f GB/s %8s\n",
               cfg.name, ms, gflops, eff_bw, status);
    }

    printf("\nFP32 Peak (theoretical): %.0f GFLOPS\n",
           (double)prop.multiProcessorCount * 64 * 2 *
           prop.clockRate / 1e6);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);

    return 0;
}
