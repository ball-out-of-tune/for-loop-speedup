// 测试: shared memory 重复读会被编译器优化吗?
#define BLOCK 256
__global__ void reduce_shared(float *out) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    // 读两次 smem[tid+stride]
    if (smem[tid] < smem[tid + 128]) {
        smem[tid] = smem[tid + 128];
    }
}
int main() { reduce_shared<<<1,BLOCK>>>(0); cudaDeviceSynchronize(); return 0; }
