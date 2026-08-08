#define BLOCK 256
__global__ void reduce_complex(float *out) {
    __shared__ float smem_max[BLOCK];
    __shared__ float smem_exp[BLOCK];
    int tid = threadIdx.x;
    // 你的真实代码: 每个 shared 变量在分支里用了多次
    if (smem_max[tid] < smem_max[tid + 64]) {
        smem_exp[tid] = smem_exp[tid] * expf(smem_max[tid] - smem_max[tid + 64])
                      + smem_exp[tid + 64];
        smem_max[tid] = smem_max[tid + 64];
    }
}
int main() { reduce_complex<<<1,BLOCK>>>(0); cudaDeviceSynchronize(); return 0; }
