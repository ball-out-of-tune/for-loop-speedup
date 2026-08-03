/**
 * Isolate: register cache vs online softmax — which one matters more?
 *
 * Same hidden=4096 (16 items/thread, all compile-time constants, all 0 spill):
 *   A: naive + x_reg    = 3-phase, 1R+1W  (register cache alone)
 *   B: online + x_reg   = online, 1R+1W   (both optimizations)
 *   C: online + no_reg  = online, 2R+1W   (online alone)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 1000
#define BLOCK 256
#define HIDDEN 4096
#define ITEMS 16        // compile-time constant!
#define WARMUP 10
#define RUNS 30

// --- A: register cache alone (naive 3-phase from regs, 1R+1W) ---
__global__ void A_naive_reg(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float x_reg[ITEMS];
    #pragma unroll
    for(int i=0;i<ITEMS;i++)x_reg[i]=xr[t+i*tt];

    float lm=-INFINITY;
    #pragma unroll
    for(int i=0;i<ITEMS;i++){if(x_reg[i]>lm)lm=x_reg[i];}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];

    float ls=0;
    #pragma unroll
    for(int i=0;i<ITEMS;i++)ls+=expf(x_reg[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];

    #pragma unroll
    for(int i=0;i<ITEMS;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- B: both (online + register cache, 1R+1W) ---
__global__ void B_online_reg(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float x_reg[ITEMS];
    #pragma unroll
    for(int i=0;i<ITEMS;i++)x_reg[i]=xr[t+i*tt];

    float m=-INFINITY,d=0;
    #pragma unroll
    for(int i=0;i<ITEMS;i++){float vv=x_reg[i];
        if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];

    #pragma unroll
    for(int i=0;i<ITEMS;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- C: online alone (online, 2R+1W, no x_reg) ---
__global__ void C_online_noreg(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];

    float m=-INFINITY,d=0;
    for(int i=t;i<hs;i+=tt){float v=xr[i];
        if(v>m){d=d*expf(m-v)+1.0f;m=v;}else{d+=expf(v-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];

    for(int i=t;i<hs;i+=tt)or_[i]=expf(xr[i]-rm)/rs;
}

float ms(cudaEvent_t s,cudaEvent_t e){float x;cudaEventElapsedTime(&x,s,e);return x;}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    printf("===== Isolate: register cache vs online =====\n");
    printf("  hidden=%d, items=%d (compile-time constant), spill assumed 0\n\n",HIDDEN,ITEMS);

    long long tb=(long long)N_ROWS*HIDDEN*4;
    float*hx=(float*)malloc(tb);
    for(long long i=0;i<(long long)N_ROWS*HIDDEN;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
    float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
    chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
    dim3 g(N_ROWS),b(BLOCK);

    cudaEvent_t st,sp;chk(cudaEventCreate(&st),"ev");chk(cudaEventCreate(&sp),"ev");

    typedef void(*kfn)(const float*,float*,int,int);
    kfn ks[]={A_naive_reg, B_online_reg, C_online_noreg};
    const char*ns[]={
        "A: naive + x_reg  (1R+1W)",
        "B: online + x_reg (1R+1W)",
        "C: online, no_reg (2R+1W)"
    };
    float traffics[]={tb*2.0f, tb*2.0f, tb*3.0f};

    for(int k=0;k<3;k++){
        for(int i=0;i<WARMUP;i++)ks[k]<<<g,b>>>(dx,do_,N_ROWS,HIDDEN);
        chk(cudaDeviceSynchronize(),"wu");
        float best=1e9;
        for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);ks[k]<<<g,b>>>(dx,do_,N_ROWS,HIDDEN);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}
        float bw=traffics[k]/(best/1000.0f)/1e9f;
        printf("%-30s %7.4f ms  BW %6.1f GB/s (%5.1f%%)\n",ns[k],best,bw,bw/192.0f*100);
    }

    printf("\n===== Answer =====\n");
    printf("  A vs C: register cache vs online — which matters more?\n");
    printf("  A vs B: does online help when data is already in registers?\n");

    cudaFree(dx);cudaFree(do_);free(hx);
    cudaEventDestroy(st);cudaEventDestroy(sp);
    return 0;
}
