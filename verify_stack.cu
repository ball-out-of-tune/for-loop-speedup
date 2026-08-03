/** Verify: B's stack frame, occupancy, and L1 behavior */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 120  /* full wave: 20 SM x 6 blocks */
#define BLOCK 256
#define HIDDEN 32768
#define IT 128

__global__ void B_online_reg(const float*x,float*o,int nr,int hs){
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

void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    long long tb=(long long)N_ROWS*HIDDEN*4;
    float*hx=(float*)malloc(tb);
    for(long long i=0;i<(long long)N_ROWS*HIDDEN;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
    float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
    chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
    dim3 g(N_ROWS),b(BLOCK);

    printf("=== B: online + x_reg[128] ===\n");
    B_online_reg<<<g,b>>>(dx,do_,N_ROWS,HIDDEN);
    chk(cudaDeviceSynchronize(),"B");

    printf("=== C: online, no x_reg ===\n");
    C_online_noreg<<<g,b>>>(dx,do_,N_ROWS,HIDDEN);
    chk(cudaDeviceSynchronize(),"C");

    printf("Done. Check ncu for stack frame, occupancy, L1 hit.\n");
    cudaFree(dx);cudaFree(do_);free(hx);
    return 0;
}
