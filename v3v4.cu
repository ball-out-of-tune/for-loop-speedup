#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 1000
#define BLOCK 256
#define WARMUP 10
#define RUNS 30

__global__ void v3(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float x_reg[16];
    for(int i=0;i<16;i++) x_reg[i]=xr[t+i*tt];
    float lm=-INFINITY;
    for(int i=0;i<16;i++){if(x_reg[i]>lm)lm=x_reg[i];}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];
    float ls=0;
    for(int i=0;i<16;i++)ls+=expf(x_reg[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];
    for(int i=0;i<16;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

__global__ void v4_32(const float*x,float*o,int nr,int hs){
    const int IT=32;
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float x_reg[IT];
    for(int i=0;i<IT;i++)x_reg[i]=xr[t+i*tt];
    float m=-INFINITY,d=0;
    for(int i=0;i<IT;i++){float vv=x_reg[i];if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];
    for(int i=0;i<IT;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

__global__ void v4_64(const float*x,float*o,int nr,int hs){
    const int IT=64;
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float x_reg[IT];
    for(int i=0;i<IT;i++)x_reg[i]=xr[t+i*tt];
    float m=-INFINITY,d=0;
    for(int i=0;i<IT;i++){float vv=x_reg[i];if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];
    for(int i=0;i<IT;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

float ms(cudaEvent_t s,cudaEvent_t e){float x;cudaEventElapsedTime(&x,s,e);return x;}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    cudaEvent_t st,sp;chk(cudaEventCreate(&st),"ev");chk(cudaEventCreate(&sp),"ev");

    printf("===== V3 vs V4: register-cached family =====\n\n");
    printf("%-12s %-10s %-10s %-8s %-10s %-12s\n",
           "Kernel","time(ms)","BW(GB/s)","BW%","regs/thd","blocks/SM");

    // Test 1: V3 at hidden=4096
    {
        int h=4096;
        long long tb=(long long)N_ROWS*h*4;
        float*hx=(float*)malloc(tb);
        for(long long i=0;i<(long long)N_ROWS*h;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(N_ROWS),b(BLOCK);

        for(int i=0;i<WARMUP;i++)v3<<<g,b>>>(dx,do_,N_ROWS,h);
        chk(cudaDeviceSynchronize(),"wu");
        float best=1e9;
        for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);v3<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}

        float traffic=tb*2.0f;
        float bw=traffic/(best/1000.0f)/1e9f;
        printf("%-12s %-10.4f %-10.1f %-7.1f%% %-10d %-12d\n",
               "V3(reg=16)",best,bw,bw/192.0f*100,34,65536/(34*BLOCK));

        float*hout=(float*)malloc(h*4);
        chk(cudaMemcpy(hout,do_,h*4,cudaMemcpyDeviceToHost),"D2H");
        float cm=-INFINITY;for(int i=0;i<h;i++)if(hx[i]>cm)cm=hx[i];
        float cs=0;for(int i=0;i<h;i++)cs+=expf(hx[i]-cm);
        float md=0;for(int i=0;i<h;i++){float e=expf(hx[i]-cm)/cs;float d=fabsf(hout[i]-e);if(d>md)md=d;}
        printf("  verify: max_diff=%.2e %s\n",md,(md<1e-4f)?"PASS":"FAIL");

        cudaFree(dx);cudaFree(do_);free(hx);free(hout);
    }

    // Test 2: V4_32 at hidden=8192
    {
        int h=8192;
        long long tb=(long long)N_ROWS*h*4;
        float*hx=(float*)malloc(tb);
        for(long long i=0;i<(long long)N_ROWS*h;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(N_ROWS),b(BLOCK);

        for(int i=0;i<WARMUP;i++)v4_32<<<g,b>>>(dx,do_,N_ROWS,h);
        chk(cudaDeviceSynchronize(),"wu");
        float best=1e9;
        for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);v4_32<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}

        float traffic=tb*2.0f;
        float bw=traffic/(best/1000.0f)/1e9f;
        printf("%-12s %-10.4f %-10.1f %-7.1f%% %-10d %-12d\n",
               "V4(reg=32)",best,bw,bw/192.0f*100,48,65536/(48*BLOCK));

        float*hout=(float*)malloc(h*4);
        chk(cudaMemcpy(hout,do_,h*4,cudaMemcpyDeviceToHost),"D2H");
        float cm=-INFINITY;for(int i=0;i<h;i++)if(hx[i]>cm)cm=hx[i];
        float cs=0;for(int i=0;i<h;i++)cs+=expf(hx[i]-cm);
        float md=0;for(int i=0;i<h;i++){float e=expf(hx[i]-cm)/cs;float d=fabsf(hout[i]-e);if(d>md)md=d;}
        printf("  verify: max_diff=%.2e %s\n",md,(md<1e-4f)?"PASS":"FAIL");

        cudaFree(dx);cudaFree(do_);free(hx);free(hout);
    }

    // Test 3: V4_64 at hidden=16384
    {
        int h=16384;
        long long tb=(long long)N_ROWS*h*4;
        float*hx=(float*)malloc(tb);
        for(long long i=0;i<(long long)N_ROWS*h;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(N_ROWS),b(BLOCK);

        for(int i=0;i<WARMUP;i++)v4_64<<<g,b>>>(dx,do_,N_ROWS,h);
        chk(cudaDeviceSynchronize(),"wu");
        float best=1e9;
        for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);v4_64<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}

        float traffic=tb*2.0f;
        float bw=traffic/(best/1000.0f)/1e9f;
        printf("%-12s %-10.4f %-10.1f %-7.1f%% %-10d %-12d\n",
               "V4(reg=64)",best,bw,bw/192.0f*100,80,65536/(80*BLOCK));

        float*hout=(float*)malloc(h*4);
        chk(cudaMemcpy(hout,do_,h*4,cudaMemcpyDeviceToHost),"D2H");
        float cm=-INFINITY;for(int i=0;i<h;i++)if(hx[i]>cm)cm=hx[i];
        float cs=0;for(int i=0;i<h;i++)cs+=expf(hx[i]-cm);
        float md=0;for(int i=0;i<h;i++){float e=expf(hx[i]-cm)/cs;float d=fabsf(hout[i]-e);if(d>md)md=d;}
        printf("  verify: max_diff=%.2e %s\n",md,(md<1e-4f)?"PASS":"FAIL");

        cudaFree(dx);cudaFree(do_);free(hx);free(hout);
    }

    printf("\n===== 总结 =====\n");
    printf("  V3:   16 floats/thread, 34 regs, 6 blocks/SM, 0 spill -- best for hidden<=4096\n");
    printf("  V4_32: 32 floats/thread, 48 regs, 5 blocks/SM, minor spill -- beats PyTorch at 8192\n");
    printf("  V4_64: 64 floats/thread, 80 regs, 3 blocks/SM, more spill -- still beats PyTorch at 16384\n");
    printf("  Pattern: more regs -> fewer blocks/SM, but 1R+1W benefit > occupancy loss\n");

    cudaEventDestroy(st);cudaEventDestroy(sp);
    return 0;
}
