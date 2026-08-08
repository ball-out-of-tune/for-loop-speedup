#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 1000
#define HIDDEN 32768
#define BLOCK 256

__device__ inline void merge_pair(float ma,float da,float mb,float db,float*m,float*dd){
    if(ma>=mb){*m=ma;*dd=da+db*expf(mb-ma);}else{*m=mb;*dd=db+da*expf(ma-mb);}
}
__global__ void softmax_v1(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float lm=-INFINITY;
    for(int i=t;i<hs;i+=tt){float v=xr[i];if(v>lm)lm=v;}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;
    o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];
    float ls=0;
    for(int i=t;i<hs;i+=tt)ls+=expf(xr[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];
    v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];
    for(int i=t;i<hs;i+=tt)or_[i]=expf(xr[i]-rm)/rs;
}
__global__ void softmax_v2(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float m=-INFINITY,d=0;
    for(int i=t;i<hs;i+=tt){float vv=xr[i];if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float nm,nd;merge_pair(sm[t],sd[t],sm[t+st],sd[t+st],&nm,&nd);sm[t]=nm;sd[t]=nd;}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){
        float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        float nm,nd;merge_pair(m,d,om,od,&nm,&nd);m=nm;d=nd;
    }
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];
    for(int i=t;i<hs;i+=tt)or_[i]=expf(xr[i]-rm)/rs;
}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}
int main(){
    int nr=N_ROWS,hs=HIDDEN;
    long long tb=(long long)nr*hs*4;
    printf("N_ROWS=%d HIDDEN=%d data=%.1f MB  (120 rows = 1 full wave, 6 blocks per SM)\n",nr,hs,tb/1e6f);
    printf("6 blocks/SM x 128 KB/row = 768 KB/SM > 128 KB L1 -> L1 will collapse\n");
    printf("120 rows x 128 KB = 15 MB total >> 2 MB L2 -> L2 will also be stressed\n\n");
    float*hx=(float*)malloc(tb);
    for(long long i=0;i<(long long)nr*hs;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
    float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
    chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
    dim3 g(nr),b(BLOCK);
    printf("V1...\n");softmax_v1<<<g,b>>>(dx,do_,nr,hs);chk(cudaDeviceSynchronize(),"v1");
    printf("V2...\n");softmax_v2<<<g,b>>>(dx,do_,nr,hs);chk(cudaDeviceSynchronize(),"v2");
    printf("OK\n");cudaFree(dx);cudaFree(do_);free(hx);return 0;
}
