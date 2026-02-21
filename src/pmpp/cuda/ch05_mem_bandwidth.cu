/**
 * PMPP Chapter 5 - Memory Hierarchy Bandwidth Benchmark
 *
 * 测量 GPU 不同内存层级的有效带宽:
 *   1. Global Memory 读带宽 (合并 vs 非合并)
 *   2. Shared Memory 读带宽
 *   3. Constant Memory 读带宽
 *   4. Register 读带宽 (作为基线)
 *
 * 编译: nvcc -O3 -arch=sm_80 -o ch05_mem_bandwidth ch05_mem_bandwidth.cu
 * 运行: ./ch05_mem_bandwidth
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define N (1 << 24)       // 16M elements
#define ITERATIONS 100

// ============================================================
// Test 1: Global Memory - Coalesced Access (连续读取)
// 相邻线程读相邻地址 → 合并为少量 memory transactions
// ============================================================

__global__ void global_coalesced(const float* __restrict__ input,
                                  float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    if (idx < n) output[idx] = sum;
}

// ============================================================
// Test 2: Global Memory - Strided Access (跳跃读取)
// 相邻线程读取间隔 stride 的地址 → 无法合并
// ============================================================

__global__ void global_strided(const float* __restrict__ input,
                                float* __restrict__ output, int n, int access_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int mapped = (idx * access_stride) % n;
        output[idx] = input[mapped];
    }
}

// ============================================================
// Test 3: Shared Memory - 从 Global 加载到 Shared, 再密集读取
// ============================================================

__global__ void shared_mem_test(const float* __restrict__ input,
                                 float* __restrict__ output, int n) {
    __shared__ float smem[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    if (idx < n) smem[tx] = input[idx];
    __syncthreads();

    float sum = 0.0f;
    // 从 Shared Memory 反复读取, 模拟高复用场景
    for (int iter = 0; iter < 32; iter++) {
        sum += smem[(tx + iter) % 256];
    }

    if (idx < n) output[idx] = sum;
}

// ============================================================
// Test 4: Constant Memory
// ============================================================

__constant__ float const_data[1024];

__global__ void constant_mem_test(float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += const_data[i % 1024];
        }
        output[idx] = sum;
    }
}

// ============================================================
// Test 5: Register-only (无内存访问, 纯计算基线)
// ============================================================

__global__ void register_only(float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)idx;
    for (int i = 0; i < 32; i++) {
        val = val * 1.01f + 0.01f;
    }
    if (idx < n) output[idx] = val;
}

// ============================================================
// Benchmark 框架
// ============================================================

float benchmark(void (*setup)(const float*, float*, int),
                const float* d_in, float* d_out, int n, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 3; i++) setup(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) setup(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

void run_coalesced(const float* d_in, float* d_out, int n) {
    global_coalesced<<<(n+255)/256, 256>>>(d_in, d_out, n);
}
void run_strided(const float* d_in, float* d_out, int n) {
    global_strided<<<(n+255)/256, 256>>>(d_in, d_out, n, 32);
}
void run_shared(const float* d_in, float* d_out, int n) {
    shared_mem_test<<<(n+255)/256, 256>>>(d_in, d_out, n);
}
void run_constant(const float* d_in, float* d_out, int n) {
    constant_mem_test<<<(n+255)/256, 256>>>(d_out, n);
}
void run_register(const float* d_in, float* d_out, int n) {
    register_only<<<(n+255)/256, 256>>>(d_out, n);
}

int main() {
    printf("=== PMPP Ch05: Memory Hierarchy Bandwidth Benchmark ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Global Memory: %.1f GB, Bandwidth: theoretical ~%.0f GB/s\n",
           prop.totalGlobalMem / 1e9,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Test size: %d elements (%.1f MB)\n\n", N, N * 4.0 / 1e6);

    float* d_in;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // 初始化
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化 Constant Memory
    float h_const[1024];
    for (int i = 0; i < 1024; i++) h_const[i] = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(const_data, h_const, 1024 * sizeof(float)));

    printf("%-35s %10s %12s\n", "Test", "Time(ms)", "Eff.BW(GB/s)");
    printf("%s\n", std::string(60, '-').c_str());

    struct {
        const char* name;
        void (*func)(const float*, float*, int);
        double bytes_per_elem;
    } tests[] = {
        {"Global Memory (coalesced)",   run_coalesced, 8.0},   // read + write
        {"Global Memory (stride=32)",   run_strided,   8.0},
        {"Shared Memory (32x reuse)",   run_shared,    8.0},
        {"Constant Memory (broadcast)", run_constant,  4.0},
        {"Register-only (baseline)",    run_register,  4.0},
    };

    for (auto& t : tests) {
        float ms = benchmark(t.func, d_in, d_out, N, ITERATIONS);
        double bw = N * t.bytes_per_elem / (ms / 1000.0) / 1e9;
        printf("%-35s %8.3f ms %10.1f\n", t.name, ms, bw);
    }

    printf("\n说明:\n");
    printf("  - Coalesced: 相邻线程读相邻地址, 充分利用内存带宽\n");
    printf("  - Strided:   相邻线程跳读, 带宽利用率大幅下降\n");
    printf("  - Shared:    数据在 Shared Memory 中被复用 32 次\n");
    printf("  - Constant:  所有线程读同一地址, 利用广播机制\n");
    printf("  - Register:  无内存访问, 纯计算吞吐量基线\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_data);
    return 0;
}
