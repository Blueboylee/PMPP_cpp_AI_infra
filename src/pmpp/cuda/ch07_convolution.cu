/**
 * PMPP Chapter 7 - Convolution
 *
 * 本文件包含 1D 与 2D 卷积的递进实现:
 *   1D: 基础版本、常量内存（卷积核）、Tiled + Halo
 *   2D: 基础版本、常量内存、Tiled 2D + Halo
 *
 * 编译: nvcc -O3 -arch=sm_80 -o ch07_convolution ch07_convolution.cu
 * 运行: ./ch07_convolution [1d|2d]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// 1D 卷积
// 输出: out[i] = sum_n (in[i+n] * mask[n]), n = 0..maskLen-1
// 边界: 越界输入视为 0 (clamp to zero)
// ============================================================

#define MAX_MASK_1D 32

// 版本 1: 1D 基础卷积，卷积核在全局内存
__global__ void conv1d_basic(const float* in, float* out, const float* mask,
                             int n, int maskLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < maskLen; j++) {
        int idx = i + j;
        float val = (idx < n) ? in[idx] : 0.0f;
        sum += val * mask[j];
    }
    out[i] = sum;
}

// 版本 2: 1D 卷积，卷积核在常量内存（广播、缓存）
__constant__ float c_mask_1d[MAX_MASK_1D];

__global__ void conv1d_constant(const float* in, float* out,
                                 int n, int maskLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < maskLen; j++) {
        int idx = i + j;
        float val = (idx < n) ? in[idx] : 0.0f;
        sum += val * c_mask_1d[j];
    }
    out[i] = sum;
}

// 版本 3: 1D Tiled 卷积，带 Halo 加载
#define TILE_1D 256
#define HALO_1D (MAX_MASK_1D - 1)

__global__ void conv1d_tiled(const float* in, float* out,
                             int n, int maskLen) {
    __shared__ float tile[TILE_1D + HALO_1D];

    int tx = threadIdx.x;
    int base = blockIdx.x * TILE_1D;

    // 协作加载: 每个线程加载 1 个元素（含右侧 halo）
    int loadIdx = base + tx;
    tile[tx] = (loadIdx < n) ? in[loadIdx] : 0.0f;
    if (tx < HALO_1D) {
        int haloIdx = base + TILE_1D + tx;
        tile[TILE_1D + tx] = (haloIdx < n) ? in[haloIdx] : 0.0f;
    }
    __syncthreads();

    int i = base + tx;
    if (i >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < maskLen; j++) {
        sum += tile[tx + j] * c_mask_1d[j];
    }
    out[i] = sum;
}

// ============================================================
// 2D 卷积
// 输出: out[r][c] = sum_{dr,dc} in[r+dr][c+dc] * mask[dr][dc]
// mask 尺寸: maskRows x maskCols
// ============================================================

#define MAX_MASK_2D 16
#define TILE_2D 16
#define HALO_2D_R 1
#define HALO_2D_C 1
#define TILE_2D_PAD_R (TILE_2D + 2 * HALO_2D_R)
#define TILE_2D_PAD_C (TILE_2D + 2 * HALO_2D_C)

// 版本 1: 2D 基础卷积
__global__ void conv2d_basic(const float* in, float* out,
                              int rows, int cols,
                              const float* mask, int maskRows, int maskCols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    float sum = 0.0f;
    for (int dr = 0; dr < maskRows; dr++) {
        for (int dc = 0; dc < maskCols; dc++) {
            int ir = r + dr;
            int ic = c + dc;
            float val = 0.0f;
            if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                val = in[ir * cols + ic];
            sum += val * mask[dr * maskCols + dc];
        }
    }
    out[r * cols + c] = sum;
}

// 常量内存存放 2D 卷积核（小尺寸）
__constant__ float c_mask_2d[MAX_MASK_2D * MAX_MASK_2D];

__global__ void conv2d_constant(const float* in, float* out,
                                 int rows, int cols,
                                 int maskRows, int maskCols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    float sum = 0.0f;
    for (int dr = 0; dr < maskRows; dr++) {
        for (int dc = 0; dc < maskCols; dc++) {
            int ir = r + dr;
            int ic = c + dc;
            float val = 0.0f;
            if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                val = in[ir * cols + ic];
            sum += val * c_mask_2d[dr * maskCols + dc];
        }
    }
    out[r * cols + c] = sum;
}

// 版本 3: 2D Tiled 卷积，带 Halo（3x3 为例：halo=1 四周各 1）
__global__ void conv2d_tiled(const float* in, float* out,
                              int rows, int cols,
                              int maskRows, int maskCols) {
    __shared__ float tile[TILE_2D_PAD_R][TILE_2D_PAD_C];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gr = blockIdx.y * TILE_2D + ty - HALO_2D_R;
    int gc = blockIdx.x * TILE_2D + tx - HALO_2D_C;
    tile[ty][tx] = (gr >= 0 && gr < rows && gc >= 0 && gc < cols) ? in[gr * cols + gc] : 0.0f;
    __syncthreads();

    int outRow = blockIdx.y * TILE_2D + (ty - HALO_2D_R);
    int outCol = blockIdx.x * TILE_2D + (tx - HALO_2D_C);
    if (ty >= HALO_2D_R && ty < TILE_2D_PAD_R - HALO_2D_R && tx >= HALO_2D_C && tx < TILE_2D_PAD_C - HALO_2D_C
        && outRow >= 0 && outRow < rows && outCol >= 0 && outCol < cols) {
        float sum = 0.0f;
        for (int dr = 0; dr < maskRows; dr++) {
            for (int dc = 0; dc < maskCols; dc++) {
                sum += tile[ty - HALO_2D_R + dr][tx - HALO_2D_C + dc] * c_mask_2d[dr * maskCols + dc];
            }
        }
        out[outRow * cols + outCol] = sum;
    }
}

// ============================================================
// CPU 参考实现
// ============================================================

void conv1d_cpu(const float* in, float* out, const float* mask,
                int n, int maskLen) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < maskLen; j++) {
            int idx = i + j;
            float val = (idx < n) ? in[idx] : 0.0f;
            sum += val * mask[j];
        }
        out[i] = sum;
    }
}

void conv2d_cpu(const float* in, float* out, int rows, int cols,
                const float* mask, int maskRows, int maskCols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            for (int dr = 0; dr < maskRows; dr++) {
                for (int dc = 0; dc < maskCols; dc++) {
                    int ir = r + dr;
                    int ic = c + dc;
                    float val = 0.0f;
                    if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                        val = in[ir * cols + ic];
                    sum += val * mask[dr * maskCols + dc];
                }
            }
            out[r * cols + c] = sum;
        }
    }
}

// ============================================================
// Benchmark 辅助
// ============================================================

static float benchmark_1d(void (*kernel)(const float*, float*, const float*, int, int),
                          const float* d_in, float* d_out, const float* d_mask,
                          int n, int maskLen, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 3; i++)
        kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, d_mask, n, maskLen);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, d_mask, n, maskLen);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

static float benchmark_1d_const(void (*kernel)(const float*, float*, int, int),
                                 const float* d_in, float* d_out,
                                 int n, int maskLen, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 3; i++)
        kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n, maskLen);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n, maskLen);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

static float benchmark_1d_tiled(const float* d_in, float* d_out,
                                 int n, int maskLen, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int blocks = (n + TILE_1D - 1) / TILE_1D;
    for (int i = 0; i < 3; i++)
        conv1d_tiled<<<blocks, TILE_1D>>>(d_in, d_out, n, maskLen);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        conv1d_tiled<<<blocks, TILE_1D>>>(d_in, d_out, n, maskLen);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

int main(int argc, char** argv) {
    int run1d = 1;
    int run2d = 1;
    if (argc > 1) {
        if (strcmp(argv[1], "1d") == 0) run2d = 0;
        else if (strcmp(argv[1], "2d") == 0) run1d = 0;
    }

    const int n1d = 1 << 24;
    const int maskLen = 7;
    const int iters = 50;

    float h_mask_1d[MAX_MASK_1D];
    for (int i = 0; i < maskLen; i++) h_mask_1d[i] = 1.0f / maskLen;

    if (run1d) {
        printf("=== PMPP Ch07: 1D Convolution ===\n");
        printf("N = %d, maskLen = %d, iters = %d\n\n", n1d, maskLen, iters);

        float* d_in, *d_out, *d_mask;
        CUDA_CHECK(cudaMalloc(&d_in, n1d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n1d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, maskLen * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask_1d, maskLen * sizeof(float), cudaMemcpyHostToDevice));

        float* h_in = (float*)malloc(n1d * sizeof(float));
        for (int i = 0; i < n1d; i++) h_in[i] = (float)(i % 256) / 255.0f;
        CUDA_CHECK(cudaMemcpy(d_in, h_in, n1d * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpyToSymbol(c_mask_1d, h_mask_1d, maskLen * sizeof(float)));

        float t_basic = benchmark_1d(conv1d_basic, d_in, d_out, d_mask, n1d, maskLen, iters);
        float t_const = benchmark_1d_const(conv1d_constant, d_in, d_out, n1d, maskLen, iters);
        float t_tiled = benchmark_1d_tiled(d_in, d_out, n1d, maskLen, iters);

        printf("%-30s %10.3f ms  (1.00x)\n", "1D Basic (global mask)", t_basic);
        printf("%-30s %10.3f ms  (%.2fx)\n", "1D Constant memory", t_const, t_basic / t_const);
        printf("%-30s %10.3f ms  (%.2fx)\n", "1D Tiled + Halo", t_tiled, t_basic / t_tiled);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_mask));
        free(h_in);
        printf("\n");
    }

    if (run2d) {
        const int rows = 1024, cols = 1024;
        const int maskRows = 3, maskCols = 3;
        float h_mask_2d[MAX_MASK_2D * MAX_MASK_2D];
        for (int i = 0; i < maskRows * maskCols; i++) h_mask_2d[i] = 1.0f / (maskRows * maskCols);

        printf("=== PMPP Ch07: 2D Convolution ===\n");
        printf("Image %d x %d, mask %d x %d, iters = %d\n\n", rows, cols, maskRows, maskCols, iters);

        size_t imgBytes = (size_t)rows * cols * sizeof(float);
        float* d_in, *d_out, *d_mask;
        CUDA_CHECK(cudaMalloc(&d_in, imgBytes));
        CUDA_CHECK(cudaMalloc(&d_out, imgBytes));
        CUDA_CHECK(cudaMalloc(&d_mask, maskRows * maskCols * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask_2d, maskRows * maskCols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_mask_2d, h_mask_2d, maskRows * maskCols * sizeof(float)));

        float* h_in = (float*)malloc(imgBytes);
        for (int i = 0; i < rows * cols; i++) h_in[i] = (float)(i % 256) / 255.0f;
        CUDA_CHECK(cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((cols + 15) / 16, (rows + 15) / 16);
        dim3 blockTiled(TILE_2D_PAD_C, TILE_2D_PAD_R);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int i = 0; i < 3; i++) {
            conv2d_basic<<<grid, block>>>(d_in, d_out, rows, cols, d_mask, maskRows, maskCols);
            conv2d_constant<<<grid, block>>>(d_in, d_out, rows, cols, maskRows, maskCols);
            conv2d_tiled<<<grid, blockTiled>>>(d_in, d_out, rows, cols, maskRows, maskCols);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) conv2d_basic<<<grid, block>>>(d_in, d_out, rows, cols, d_mask, maskRows, maskCols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_basic;
        CUDA_CHECK(cudaEventElapsedTime(&t_basic, start, stop));
        t_basic /= iters;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) conv2d_constant<<<grid, block>>>(d_in, d_out, rows, cols, maskRows, maskCols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_const;
        CUDA_CHECK(cudaEventElapsedTime(&t_const, start, stop));
        t_const /= iters;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) conv2d_tiled<<<grid, blockTiled>>>(d_in, d_out, rows, cols, maskRows, maskCols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t_tiled;
        CUDA_CHECK(cudaEventElapsedTime(&t_tiled, start, stop));
        t_tiled /= iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        printf("%-30s %10.3f ms  (1.00x)\n", "2D Basic (global mask)", t_basic);
        printf("%-30s %10.3f ms  (%.2fx)\n", "2D Constant memory", t_const, t_basic / t_const);
        printf("%-30s %10.3f ms  (%.2fx)\n", "2D Tiled + Halo", t_tiled, t_basic / t_tiled);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_mask));
        free(h_in);
    }

    return 0;
}
