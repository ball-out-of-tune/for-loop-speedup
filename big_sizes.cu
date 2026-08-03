/** Register cache at hidden=32768 (128 items/thread) and 65536 (256 items/thread) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 1000
#define BLOCK 256
#define WARMUP 5
#define RUNS 15

// --- A: naive reg-cached, ITEMS=128 (hidden=32768) ---
__global__ void A_naive_128(const float*x,float*o,int nr,int hs){
    const int IT=128;
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float x_reg[IT];
    for(int i=0;i<IT;i++)x_reg[i]=xr[t+i*tt];

    float lm=-INFINITY;
    for(int i=0;i<IT;i++){if(x_reg[i]>lm)lm=x_reg[i];}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];

    float ls=0;
    for(int i=0;i<IT;i++)ls+=expf(x_reg[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];

    for(int i=0;i<IT;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- B: online reg-cached, ITEMS=128 ---
__global__ void B_online_128(const float*x,float*o,int nr,int hs){
    const int IT=128;
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float x_reg[IT];
    for(int i=0;i<IT;i++)x_reg[i]=xr[t+i*tt];

    float m=-INFINITY,d=0;
    for(int i=0;i<IT;i++){float vv=x_reg[i];
        if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];

    for(int i=0;i<IT;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- C: online no-reg (2R+1W) — generic, works at any size ---
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

// --- A: naive reg-cached, ITEMS=256 (hidden=65536) ---
__global__ void A_naive_256(const float*x,float*o,int nr,int hs){
    const int IT=256;
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float x_reg[IT];
    for(int i=0;i<IT;i++)x_reg[i]=xr[t+i*tt];

    float lm=-INFINITY;
    for(int i=0;i<IT;i++){if(x_reg[i]>lm)lm=x_reg[i];}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];

    float ls=0;
    for(int i=0;i<IT;i++)ls+=expf(x_reg[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];

    for(int i=0;i<IT;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

float ms(cudaEvent_t s,cudaEvent_t e){float x;cudaEventElapsedTime(&x,s,e);return x;}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    cudaEvent_t st,sp;chk(cudaEventCreate(&st),"ev");chk(cudaEventCreate(&sp),"ev");

    struct{int hidden; void(*kA)(const float*,float*,int,int);
           void(*kB)(const float*,float*,int,int);
           void(*kC)(const float*,float*,int,int); const char*label;} tests[]={
        {32768, A_naive_128,  B_online_128,  C_online_noreg, "ITEMS=128 (hidden=32768)"},
        {65536, A_naive_256,  NULL,           C_online_noreg, "ITEMS=256 (hidden=65536)"},
    };

    printf("===== Big sizes: register cache vs online =====\n\n");

    for(int t=0;t<2;t++){
        int h=tests[t].hidden;
        long long tb=(long long)N_ROWS*h*4;
        printf("--- %s | data=%.0f MB ---\n",tests[t].label,tb/1e6f);

        float*hx=(float*)malloc(tb);
        for(long long i=0;i<(long long)N_ROWS*h;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(N_ROWS),b(BLOCK);

        // A: naive reg
        {
            for(int i=0;i<WARMUP;i++)tests[t].kA<<<g,b>>>(dx,do_,N_ROWS,h);
            chk(cudaDeviceSynchronize(),"wuA");
            float best=1e9;
            for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);tests[t].kA<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}
            float bw=tb*2.0f/(best/1000.0f)/1e9f;
            printf("  A: naive + reg   %8.4f ms  BW %6.1f GB/s (%5.1f%%)  1R+1W\n",best,bw,bw/192.0f*100);
        }

        // B: online reg (skip for 256 — will be worse)
        if(tests[t].kB){
            for(int i=0;i<WARMUP;i++)tests[t].kB<<<g,b>>>(dx,do_,N_ROWS,h);
            chk(cudaDeviceSynchronize(),"wuB");
            float best=1e9;
            for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);tests[t].kB<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}
            float bw=tb*2.0f/(best/1000.0f)/1e9f;
            printf("  B: online+ reg   %8.4f ms  BW %6.1f GB/s (%5.1f%%)  1R+1W\n",best,bw,bw/192.0f*100);
        } else {
            printf("  B: online+ reg   SKIP (256 floats won't fit)\n");
        }

        // C: online no-reg
        {
            for(int i=0;i<WARMUP;i++)tests[t].kC<<<g,b>>>(dx,do_,N_ROWS,h);
            chk(cudaDeviceSynchronize(),"wuC");
            float best=1e9;
            for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);tests[t].kC<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}
            float bw=tb*3.0f/(best/1000.0f)/1e9f;
            printf("  C: online no-reg %8.4f ms  BW %6.1f GB/s (%5.1f%%)  2R+1W\n",best,bw,bw/192.0f*100);
        }

        cudaFree(dx);cudaFree(do_);free(hx);
    }

    printf("\n===== vs PyTorch =====\n");
    printf("  PyTorch 32768: ~? (run: python -c \"import torch; ...\")\n");
    printf("  PyTorch 65536: ~? (probably slower than online at this size)\n");

    cudaEventDestroy(st);cudaEventDestroy(sp);
    return 0;
}
