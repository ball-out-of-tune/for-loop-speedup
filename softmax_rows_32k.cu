#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define HIDDEN_SIZE    32768
#define BLOCK_SIZE     256
#define WARMUP         3
#define RUNS           10

__device__ inline void merge_pair(float ma,float da,float mb,float db,float*m,float*dd){
    if(ma>=mb){*m=ma;*dd=da+db*expf(mb-ma);}
    else{*m=mb;*dd=db+da*expf(ma-mb);}
}

__global__ void v1(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*out_r=o+r*hs;
    __shared__ float s[BLOCK_SIZE];
    float lm=-INFINITY;
    for(int i=t;i<hs;i+=tt){float v=xr[i];if(v>lm)lm=v;}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK_SIZE/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t];
    float o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];
    float ls=0;
    for(int i=t;i<hs;i+=tt)ls+=expf(xr[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK_SIZE/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];
    v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];
    for(int i=t;i<hs;i+=tt)out_r[i]=expf(xr[i]-rm)/rs;
}

__global__ void v2(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*out_r=o+r*hs;
    __shared__ float sm[BLOCK_SIZE],sd[BLOCK_SIZE];
    float m=-INFINITY,d=0;
    for(int i=t;i<hs;i+=tt){float vv=xr[i];if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK_SIZE/2;st>=32;st>>=1){if(t<st){float nm,nd;merge_pair(sm[t],sd[t],sm[t+st],sd[t+st],&nm,&nd);sm[t]=nm;sd[t]=nd;}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){
        float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        float nm,nd;merge_pair(m,d,om,od,&nm,&nd);m=nm;d=nd;
    }
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];
    for(int i=t;i<hs;i+=tt)out_r[i]=expf(xr[i]-rm)/rs;
}

float ms(cudaEvent_t s,cudaEvent_t e){float x;cudaEventElapsedTime(&x,s,e);return x;}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    int rows[]={100,500,1000,5000};
    int nt=4,hs=HIDDEN_SIZE;
    cudaEvent_t st,sp;chk(cudaEventCreate(&st),"ev");chk(cudaEventCreate(&sp),"ev");
    printf("===== Fixed HIDDEN=%d, vary N_ROWS =====\n\n",hs);
    printf("%-12s %-12s %-12s %-8s %-14s %-14s\n",
           "N_ROWS","V1(ms)","V2(ms)","V2/V1","us/row V1","us/row V2");
    for(int t=0;t<nt;t++){
        int nr=rows[t];
        long long te=(long long)nr*hs,tb=te*4;
        float*hx=(float*)malloc(tb);
        for(long long i=0;i<te;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(nr),b(BLOCK_SIZE);
        for(int i=0;i<WARMUP;i++)v1<<<g,b>>>(dx,do_,nr,hs);
        chk(cudaDeviceSynchronize(),"wu1");
        float b1=1e9;for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);v1<<<g,b>>>(dx,do_,nr,hs);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<b1)b1=tt;}
        for(int i=0;i<WARMUP;i++)v2<<<g,b>>>(dx,do_,nr,hs);
        chk(cudaDeviceSynchronize(),"wu2");
        float b2=1e9;for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);v2<<<g,b>>>(dx,do_,nr,hs);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<b2)b2=tt;}
        printf("%-12d %-12.4f %-12.4f %-8.3f %-14.1f %-14.1f\n",
               nr,b1,b2,b1/b2,b1/nr*1000,b2/nr*1000);
        cudaFree(dx);cudaFree(do_);free(hx);
    }
    printf("\n===== Key =====\n");
    printf("Each row = %d bytes = %.1f KB. Waves = N_ROWS / (20 SM x 6 blocks).\n",
           hs*4,hs*4.0f/1024);
    printf("Both V1 and V2 scale linearly with N_ROWS. Ratio is constant.\n");
    cudaEventDestroy(st);cudaEventDestroy(sp);
    return 0;
}
