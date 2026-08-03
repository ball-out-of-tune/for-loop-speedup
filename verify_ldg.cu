/** Verify: where does MyOwn vs V4 read from for the 2nd pass? */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS 20
#define HIDDEN 8192
#define BLOCK  256
#define MY_BLOCK_SIZE 256

// MyOwn: online with 2nd pass re-reading from global memory
__global__ void softmax_myown(const float *x, float *out, int n_rows, int hidden_size) {
    int tid = threadIdx.x;
    int gap = blockIdx.x * hidden_size;
    const float* x_cur_ptr = x + gap;
    float* out_cur_ptr = out + gap;
    __shared__ float smem_max[MY_BLOCK_SIZE], smem_exp[MY_BLOCK_SIZE];
    float _max = -INFINITY, exp_sum = 0.0f;

    for (int i = tid; i < hidden_size; i += MY_BLOCK_SIZE) {
        float temp1 = x_cur_ptr[i];                  // ← LDG (phase 1)
        if (temp1 > _max) { exp_sum = exp_sum * expf(_max - temp1) + 1.0f; _max = temp1; }
        else { exp_sum += expf(temp1 - _max); }
    }
    smem_max[tid] = _max; smem_exp[tid] = exp_sum; __syncthreads();
    for (int stride = MY_BLOCK_SIZE/2; stride >= 32; stride /= 2) {
        if (tid < stride) {
            if (smem_max[tid] < smem_max[tid+stride]) {
                smem_exp[tid] = smem_exp[tid] * expf(smem_max[tid] - smem_max[tid+stride]) + smem_exp[tid+stride];
                smem_max[tid] = smem_max[tid+stride];
            } else {
                smem_exp[tid] = smem_exp[tid] + smem_exp[tid+stride] * expf(smem_max[tid+stride] - smem_max[tid]);
            }
        }
        __syncthreads();
    }
    float val = smem_exp[tid], _max2 = smem_max[tid];
    float t = __shfl_down_sync(0xffffffff, _max2, 16);
    if (_max2 < t) { val = val * expf(_max2 - t) + __shfl_down_sync(0xffffffff, val, 16); _max2 = t; }
    else { val = val + __shfl_down_sync(0xffffffff, val, 16) * expf(t - _max2); }
    t = __shfl_down_sync(0xffffffff, _max2, 8);
    if (_max2 < t) { val = val * expf(_max2 - t) + __shfl_down_sync(0xffffffff, val, 8); _max2 = t; }
    else { val = val + __shfl_down_sync(0xffffffff, val, 8) * expf(t - _max2); }
    t = __shfl_down_sync(0xffffffff, _max2, 4);
    if (_max2 < t) { val = val * expf(_max2 - t) + __shfl_down_sync(0xffffffff, val, 4); _max2 = t; }
    else { val = val + __shfl_down_sync(0xffffffff, val, 4) * expf(t - _max2); }
    t = __shfl_down_sync(0xffffffff, _max2, 2);
    if (_max2 < t) { val = val * expf(_max2 - t) + __shfl_down_sync(0xffffffff, val, 2); _max2 = t; }
    else { val = val + __shfl_down_sync(0xffffffff, val, 2) * expf(t - _max2); }
    t = __shfl_down_sync(0xffffffff, _max2, 1);
    if (_max2 < t) { val = val * expf(_max2 - t) + __shfl_down_sync(0xffffffff, val, 1); _max2 = t; }
    else { val = val + __shfl_down_sync(0xffffffff, val, 1) * expf(t - _max2); }
    if (tid == 0) { smem_max[tid] = _max2; smem_exp[tid] = val; }
    __syncthreads();
    float max_final = smem_max[0], exp_final_sum = smem_exp[0];

    for (int i = tid; i < hidden_size; i += BLOCK) {
        out_cur_ptr[i] = expf(x_cur_ptr[i] - max_final) / exp_final_sum;  // ← LDG (phase 2)!
    }
}

// V4: register-cached, no 2nd read
__global__ void softmax_v4(const float *x, float *out, int n_rows, int hidden_size) {
    const int ITEMS = 32;
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm_m[BLOCK], sm_d[BLOCK];

    float x_reg[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) x_reg[i] = xr[tid + i * tt];  // ← 32 LDG (phase 1 only)

    float m = -INFINITY, d = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) { float v = x_reg[i];
        if (v > m) { d = d * expf(m - v) + 1.0f; m = v; } else { d += expf(v - m); } }

    sm_m[tid] = m; sm_d[tid] = d; __syncthreads();
    for (int s = BLOCK/2; s >= 32; s >>= 1) {
        if (tid < s) { float am=sm_m[tid],ad=sm_d[tid],bm=sm_m[tid+s],bd=sm_d[tid+s];
            if(am>=bm){sm_m[tid]=am;sm_d[tid]=ad+bd*expf(bm-am);}
            else{sm_m[tid]=bm;sm_d[tid]=bd+ad*expf(am-bm);} } __syncthreads();
    }
    m=sm_m[tid];d=sm_d[tid];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(tid==0){sm_m[0]=m;sm_d[0]=d;}__syncthreads();
    float rm=sm_m[0],rs=sm_d[0];

    #pragma unroll
    for(int i=0;i<ITEMS;i++) out_r[tid+i*tt]=expf(x_reg[i]-rm)/rs;  // ← NO LDG! Registers!
}

void check(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    long long tb=(long long)N_ROWS*HIDDEN*4;
    float*hx=(float*)malloc(tb);
    for(long long i=0;i<(long long)N_ROWS*HIDDEN;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
    float*dx,*dout;
    check(cudaMalloc(&dx,tb),"dx");check(cudaMalloc(&dout,tb),"dout");
    check(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
    dim3 g(N_ROWS),b(BLOCK);

    printf("=== MyOwn (online, 2nd pass reads global) ===\n");
    softmax_myown<<<g,b>>>(dx,dout,N_ROWS,HIDDEN);
    check(cudaDeviceSynchronize(),"myown");

    printf("=== V4 (reg-cached, NO 2nd read) ===\n");
    softmax_v4<<<g,b>>>(dx,dout,N_ROWS,HIDDEN);
    check(cudaDeviceSynchronize(),"v4");

    printf("Done. Check ncu LDG counts above.\n");
    cudaFree(dx);cudaFree(dout);free(hx);
    return 0;
}
