"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum (float val) {
    // implement warp-level sum reduction using shuffle intrinsics
    int mask = __activemask ();
    int active = __popc(mask);
    int max_off = 1;
    while (max_off < active) max_off <<= 1;
    max_off >>= 1;
    for (int off = max_off; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    val = __shfl_sync (mask, val, 0);
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max (float val) {
    // implement warp-level max reduction using shuffle intrinsics
    int mask = __activemask ();
    int active = __popc(mask);
    int max_off = 1;
    while (max_off < active) max_off <<= 1;
    max_off >>= 1;
    for (int off = max_off; off > 0; off >>= 1)
        val = fmaxf (val, __shfl_down_sync (mask, val, off));
    val = __shfl_sync (mask, val, 0);
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // TODO: block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int tid = threadIdx.x;
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    float sum = warp_reduce_sum(val);
    if (laneId == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (laneId < numWarps) ? shared[laneId] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

# Step 4 - block_reduce_max
#include <cfloat>

__device__ float block_reduce_max (float val, float* shared) {
    // block-wide max via warp_reduce_max + shared memory
    int tid      = threadIdx.x;
    int laneId   = tid % warpSize;
    int warpId   = tid / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    float maxVal = warp_reduce_max (val);
    if (laneId == 0) {
        shared[warpId] = maxVal;
    }
    __syncthreads ();

    if (warpId == 0) {
        maxVal = (laneId < numWarps) ? shared[laneId] : -FLT_MAX;
        maxVal = warp_reduce_max (maxVal);
    }
    return maxVal;
}

# Step 5 - add_residual_kernel
__global__ void
add_residual_kernel (const float* x, const float* residual, float* out, int n) {
    // implement elementwise residual addition out[i] = x[i] + residual[i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        out[idx] = x[idx] + residual[idx];
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // Apply GELU (tanh approximation) elementwise to x, write into out
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x_val + 0.044715f * x_val * x_val * x_val)));
        out[idx] = x_val * cdf;
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel (const float* x, float* out, int n) {
    // apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_val = x[idx];
        out[idx] = x_val / (1.0f + expf(-x_val));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel (const float* gate, const float* up, float* out, int n) {
    // out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate_val = gate[idx];
        float silu_val = gate_val / (1.0f + expf(-gate_val));
        out[idx] = silu_val * up[idx];
    }
}

# Step 9 - rmsnorm_kernel
__global__ void
rmsnorm_kernel (const float* x, const float* weight, float* out, int n, float eps) {
    // Apply RMSNorm per row (one block per row)
    int tid    = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    const int rowId = blockIdx.x;
    const float *x_row = x + rowId * n;
    float *out_row = out + rowId * n;

    __shared__ float smem[32];


    float val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        val += x_row[i] * x_row[i];
    }

    float sum = block_reduce_sum (val, smem);
    if (tid == 0) {
        smem[0] = rsqrtf (sum / n + eps);
    }
    __syncthreads ();

    float inv_rms = smem[0];

    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * weight[i] * inv_rms;
    }
}

# Step 10 - layernorm_kernel
__global__ void
layernorm_kernel (const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // per-row LayerNorm using block_reduce_sum for mean and variance
    const int tid = threadIdx.x;
    const int rowId = blockIdx.x;

    const float *x_row = x + rowId * n;
    float *out_row = out + rowId * n;

    float val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        val += x_row[i];

    __shared__ float smem[32];

    float sum = block_reduce_sum(val, smem);
    if (tid == 0)
        smem[0] = sum / n;
    __syncthreads();

    float mean = smem[0];

    val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float t = x_row[i] - mean;
        val += t * t;
    }
    sum = block_reduce_sum(val, smem);
    if (tid == 0) {
        smem[0] = rsqrtf(sum / n + eps);
    }
    __syncthreads();
    
    float inv_std = smem[0];

    for (int i = tid; i < n; i += blockDim.x)
        out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
}

# Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // fuse residual addition with RMSNorm (one block per row)
    const int tid = threadIdx.x;
    const int rowId = blockIdx.x;

    x = x + rowId * n;
    residual = residual + rowId * n;
    out = out + rowId * n;
    residual_out = residual_out + rowId * n;

    __shared__ float smem[32];
    float val;
    float sum;
    float inv_rms;

    val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float r = x[i] + residual[i];
        residual_out[i] = r;
        val += r * r;
    }

    sum = block_reduce_sum(val, smem);
    if (tid == 0) {
        smem[0] = rsqrtf(sum / n + eps);
    }
    __syncthreads();
    
    inv_rms = smem[0];

    for (int i = tid; i < n; i += blockDim.x)
        out[i] = residual_out[i] * inv_rms * weight[i];
}

# Step 12 - softmax_row_kernel
__global__ void softmax_row_kernel (const float* x, float* out, int rows, int cols) {
    // implement numerically stable row-wise softmax (one block per row)
    const int rowId = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int numThreads = blockDim.x;

    extern __shared__ float smem[];
    
    const float *x_row = x + rowId * cols;
    float *out_row = out + rowId * cols;
    float val;
    float row_max;
    float row_sum;

    val = -FLT_MAX;
    for (int i = tid; i < cols; i += numThreads)
        val = fmaxf(val, x_row[i]);
    row_max = block_reduce_max(val, smem);
    if (tid == 0)
        smem[0] = row_max; 
    __syncthreads();
    row_max = smem[0];
    
    val = 0.0f;
    for (int i = tid; i < cols; i += numThreads)
        val += expf(x_row[i] - row_max);
    row_sum = block_reduce_sum(val, smem);
    if (tid == 0)
        smem[0] = row_sum;
    __syncthreads();
    row_sum = smem[0];

    for (int i = tid; i < cols; i += numThreads)
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
}

# Step 13 - causal_softmax_kernel
__global__ void causal_softmax_kernel (const float* x, float* out, int rows, int cols) {
    // numerically stable causal softmax (one block per row);
    // mask columns c > row to 0; use block_reduce_max / block_reduce_sum
    const int rowId = blockIdx.x;
    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;

    __shared__ float smem[32];

    const float *x_row = x + rowId * cols;
    float *out_row = out + rowId * cols;

    float val;
    float row_max;
    float row_sum;

    val = -FLT_MAX;
    for (int i = tid; i < cols && i <= rowId; i += numThreads)
        val = fmaxf(val, x_row[i]);
    row_max = block_reduce_max(val, smem);
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];
    __syncthreads();

    val = 0.0f;
    for (int i = tid; i < cols && i <= rowId; i += numThreads)
        val += expf(x_row[i] - row_max);
    row_sum = block_reduce_sum(val, smem);
    if (tid == 0) smem[0] = row_sum;
    __syncthreads();
    row_sum = smem[0];

    for (int i = tid; i < cols; i += numThreads) {
        out_row[i] = (i <= rowId) ? expf(x_row[i] - row_max) / row_sum : 0.0f;
    }
}

# Step 14 - embedding_lookup_kernel
__global__ void embedding_lookup_kernel (const int* token_ids, const float* weight, float* out,
                                         int seq_len, int vocab_size, int embed_dim) {
    // gather embedding vectors for each token id into out
    // token_ids: [seq_len,]
    // weight: [vocab_size, embed_dim]
    // out: [seq_len, embed_dim]
    // out[i * D + d] = weight[token_ids[i] * D + d]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < seq_len * embed_dim) {
        int token_pos = idx / embed_dim;
        int dim = idx % embed_dim;
        int token_id = token_ids[token_pos];
        out[idx] = weight[token_id * embed_dim + dim];
    }
}

# Step 15 - rope_kernel
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    // apply RoPE rotation in-place to every even/odd pair of q and k
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * head_dim / 2;

    if (idx < total) {
        int t = idx / (n_heads * head_dim / 2); // token position
        int h = (idx % (n_heads * head_dim / 2)) / (head_dim / 2);
        int pair_i = idx % (head_dim / 2);
        
        int base = (t * n_heads + h) * head_dim;
        int even = base + 2 * pair_i;
        int odd = even + 1;
        float cos = cos_table[t * head_dim / 2 + pair_i];
        float sin = sin_table[t * head_dim / 2 + pair_i];

        float q_even = q[even];
        float q_odd = q[odd];
        float k_even = k[even];
        float k_odd = k[odd];
        q[even] = q_even * cos - q_odd * sin;
        q[odd] = q_even * sin + q_odd * cos;
        k[even] = k_even * cos - k_odd * sin;
        k[odd] = k_even * sin + k_odd * cos;
    }
}

# Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;
    
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += x[m * K + k] * weight[n * K + k];
    if (bias != nullptr)
        acc += bias[n];
    out[idx] = acc;
}

# Step 17 - fused_linear_bias_gelu_kernel (not yet solved)
# TODO: implement

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

