/**
 * PMPP Chapter 6 - Warp Divergence and Memory Coalescing Experiments
 *
 * 本文件包含多个性能对比实验:
 *   1. Warp Divergence: 分支分化 vs 无分化
 *   2. Memory Coalescing: 合并访问 vs 跳跃访问 (不同 stride)
 *   3. AoS vs SoA: 结构体数组 vs 分离数组
 *   4. ILP 实验: 指令级并行度的影响
 *
 * 编译: nvcc -O3 -arch=sm_80 -o ch06_divergence ch06_divergence.cu
 * 运行: ./ch06_divergence
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define N (1 << 24)        // 16M elements
#define ITERATIONS 100

// ============================================================
// 实验 1: Warp Divergence - 分支分化
// ============================================================

__global__ void branch_divergent(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 同一 Warp 中有些线程走 if, 有些走 else → Divergence
        if (threadIdx.x % 2 == 0) {
            output[idx] = input[idx] * 2.0f;  // 偶数线程
        } else {
            output[idx] = input[idx] * 3.0f;  // 奇数线程
        }
    }
}

__global__ void branch_uniform(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 按 Warp 分组, 同一 Warp 内走同一分支 → 无 Divergence
        if ((threadIdx.x / 32) % 2 == 0) {
            output[idx] = input[idx] * 2.0f;
        } else {
            output[idx] = input[idx] * 3.0f;
        }
    }
}

__global__ void branch_predicated(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 用算术替代分支 (编译器可能优化为 predicated 指令)
        float multiplier = (threadIdx.x % 2 == 0) ? 2.0f : 3.0f;
        output[idx] = input[idx] * multiplier;
    }
}

// ============================================================
// 实验 2: Memory Coalescing - 不同访问模式
// ============================================================

__global__ void coalesced_access(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * 2.0f;  // 相邻线程读相邻地址
    }
}

__global__ void strided_access(const float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int mapped = (idx * stride) % n;
        output[idx] = input[mapped] * 2.0f;  // 相邻线程跳读
    }
}

__global__ void random_access(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 伪随机访问 (实际中可能是 hash 或索引表)
        int mapped = (idx * 1103515245 + 12345) % n;
        output[idx] = input[mapped] * 2.0f;
    }
}

// ============================================================
// 实验 3: AoS (Array of Structures) vs SoA (Structure of Arrays)
// ============================================================

struct Point3D {
    float x, y, z;
};

__global__ void aos_access(Point3D* points, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // AoS: 访问 points[idx].x, points[idx].y, points[idx].z
        // 同一 Warp 中线程访问的 x 坐标在内存中不连续 → 无法合并
        output[idx] = points[idx].x + points[idx].y + points[idx].z;
    }
}

__global__ void soa_access(const float* x_arr, const float* y_arr, const float* z_arr,
                           float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // SoA: 访问 x_arr[idx], y_arr[idx], z_arr[idx]
        // 同一 Warp 中线程访问的 x 坐标在内存中连续 → 可以合并
        output[idx] = x_arr[idx] + y_arr[idx] + z_arr[idx];
    }
}

// ============================================================
// 实验 4: ILP (Instruction-Level Parallelism)
// ============================================================

__global__ void ilp_low(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 低 ILP: 连续依赖链
        float val = input[idx];
        val = val * 1.1f + 0.1f;
        val = val * 1.1f + 0.1f;
        val = val * 1.1f + 0.1f;
        val = val * 1.1f + 0.1f;
        output[idx] = val;
    }
}

__global__ void ilp_high(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 高 ILP: 多个独立累积器, 编译器可以并行调度
        float val0 = input[idx];
        float val1 = input[idx] * 0.5f;
        float val2 = input[idx] * 0.25f;
        float val3 = input[idx] * 0.125f;
        
        val0 = val0 * 1.1f + 0.1f;
        val1 = val1 * 1.1f + 0.1f;
        val2 = val2 * 1.1f + 0.1f;
        val3 = val3 * 1.1f + 0.1f;
        
        output[idx] = val0 + val1 + val2 + val3;
    }
}

// ============================================================
// Benchmark 框架
// ============================================================

float benchmark_kernel(void (*kernel)(const float*, float*, int),
                       const float* d_in, float* d_out, int n, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++) kernel(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) kernel(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

float benchmark_kernel_stride(void (*kernel)(const float*, float*, int, int),
                              const float* d_in, float* d_out, int n, int stride, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++) kernel(d_in, d_out, n, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) kernel(d_in, d_out, n, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

int main() {
    printf("=== PMPP Ch06: Performance Considerations Experiments ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Test size: %d elements (%.1f MB)\n\n", N, N * sizeof(float) / 1e6);

    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    float* h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // ============================================================
    // 实验 1: Warp Divergence
    // ============================================================
    printf("------------------------------------------------------------\n");
    printf("Experiment 1: Warp Divergence\n");
    printf("------------------------------------------------------------\n");
    printf("%-35s %10s %12s\n", "Kernel", "Time(ms)", "Speedup");
    printf("%s\n", std::string(60, '-').c_str());

    float t_divergent = benchmark_kernel(branch_divergent, d_input, d_output, N, ITERATIONS);
    float t_uniform = benchmark_kernel(branch_uniform, d_input, d_output, N, ITERATIONS);
    float t_predicated = benchmark_kernel(branch_predicated, d_input, d_output, N, ITERATIONS);

    printf("%-35s %8.3f ms %10s\n", "Branch Divergent (50/50 split)", t_divergent, "1.00x");
    printf("%-35s %8.3f ms %10.2fx\n", "Branch Uniform (warp-aligned)", t_uniform,
           t_divergent / t_uniform);
    printf("%-35s %8.3f ms %10.2fx\n", "Branch Predicated (arithmetic)", t_predicated,
           t_divergent / t_predicated);
    printf("\n");

    // ============================================================
    // 实验 2: Memory Coalescing
    // ============================================================
    printf("------------------------------------------------------------\n");
    printf("Experiment 2: Memory Coalescing\n");
    printf("------------------------------------------------------------\n");
    printf("%-35s %10s %12s\n", "Access Pattern", "Time(ms)", "Bandwidth(GB/s)");
    printf("%s\n", std::string(60, '-').c_str());

    float t_coalesced = benchmark_kernel(coalesced_access, d_input, d_output, N, ITERATIONS);
    double bw_coalesced = N * 2 * sizeof(float) / (t_coalesced / 1000.0) / 1e9;
    printf("%-35s %8.3f ms %10.1f\n", "Coalesced (stride=1)", t_coalesced, bw_coalesced);

    for (int stride : {2, 4, 8, 16, 32}) {
        float t = benchmark_kernel_stride(strided_access, d_input, d_output, N, stride, ITERATIONS);
        double bw = N * 2 * sizeof(float) / (t / 1000.0) / 1e9;
        printf("%-35s %8.3f ms %10.1f (%.1fx slower)\n",
               ("Strided (stride=" + std::to_string(stride) + ")").c_str(),
               t, bw, t / t_coalesced);
    }

    float t_random = benchmark_kernel(random_access, d_input, d_output, N, ITERATIONS);
    double bw_random = N * 2 * sizeof(float) / (t_random / 1000.0) / 1e9;
    printf("%-35s %8.3f ms %10.1f (%.1fx slower)\n",
           "Random Access", t_random, bw_random, t_random / t_coalesced);
    printf("\n");

    // ============================================================
    // 实验 3: AoS vs SoA
    // ============================================================
    printf("------------------------------------------------------------\n");
    printf("Experiment 3: AoS vs SoA\n");
    printf("------------------------------------------------------------\n");
    printf("%-35s %10s %12s\n", "Layout", "Time(ms)", "Speedup");
    printf("%s\n", std::string(60, '-').c_str());

    Point3D* d_points;
    float* d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point3D)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));

    Point3D* h_points = (Point3D*)malloc(N * sizeof(Point3D));
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    float* h_z = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_points[i].x = h_points[i].y = h_points[i].z = 1.0f;
        h_x[i] = h_y[i] = h_z[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_points, h_points, N * sizeof(Point3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));

    float t_aos = benchmark_kernel((void(*)(const float*, float*, int))aos_access,
                                   (const float*)d_points, d_output, N, ITERATIONS);
    float t_soa = benchmark_kernel((void(*)(const float*, float*, int))soa_access,
                                    d_x, d_output, N, ITERATIONS);

    printf("%-35s %8.3f ms %10s\n", "AoS (Array of Structures)", t_aos, "1.00x");
    printf("%-35s %8.3f ms %10.2fx\n", "SoA (Structure of Arrays)", t_soa, t_aos / t_soa);
    printf("\n");

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    free(h_points);
    free(h_x);
    free(h_y);
    free(h_z);

    // ============================================================
    // 实验 4: ILP
    // ============================================================
    printf("------------------------------------------------------------\n");
    printf("Experiment 4: Instruction-Level Parallelism\n");
    printf("------------------------------------------------------------\n");
    printf("%-35s %10s %12s\n", "ILP Strategy", "Time(ms)", "Speedup");
    printf("%s\n", std::string(60, '-').c_str());

    float t_low_ilp = benchmark_kernel(ilp_low, d_input, d_output, N, ITERATIONS);
    float t_high_ilp = benchmark_kernel(ilp_high, d_input, d_output, N, ITERATIONS);

    printf("%-35s %8.3f ms %10s\n", "Low ILP (sequential deps)", t_low_ilp, "1.00x");
    printf("%-35s %8.3f ms %10.2fx\n", "High ILP (parallel ops)", t_high_ilp, t_low_ilp / t_high_ilp);
    printf("\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);

    printf("Key Takeaways:\n");
    printf("  1. Warp Divergence: Align branches to warp boundaries\n");
    printf("  2. Coalescing: Keep memory accesses contiguous within warps\n");
    printf("  3. AoS vs SoA: SoA enables coalescing for parallel access\n");
    printf("  4. ILP: Independent operations can execute in parallel\n");

    return 0;
}
