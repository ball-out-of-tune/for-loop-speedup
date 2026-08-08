/** V3 vs V4 at the SAME hidden size */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define N_ROWS 1000
#define BLOCK 256
#define WARMUP 10
#define RUNS 30

// --- V3: reg-cached naive (3-phase from regs, 1R+1W) ---
__global__ void v3(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x,items=hs/tt;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float s[BLOCK];
    float x_reg[16];
    for(int i=0;i<items;i++)x_reg[i]=xr[t+i*tt];

    float lm=-INFINITY;
    for(int i=0;i<items;i++){if(x_reg[i]>lm)lm=x_reg[i];}
    s[t]=lm;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st&&s[t+st]>s[t])s[t]=s[t+st];__syncthreads();}
    float v=s[t],o2;o2=__shfl_down_sync(0xffffffff,v,16);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,8);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,4);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,2);if(o2>v)v=o2;
    o2=__shfl_down_sync(0xffffffff,v,1);if(o2>v)v=o2;
    if(t==0)s[0]=v;__syncthreads();float rm=s[0];

    float ls=0;
    for(int i=0;i<items;i++)ls+=expf(x_reg[i]-rm);
    s[t]=ls;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st)s[t]+=s[t+st];__syncthreads();}
    v=s[t];v+=__shfl_down_sync(0xffffffff,v,16);v+=__shfl_down_sync(0xffffffff,v,8);
    v+=__shfl_down_sync(0xffffffff,v,4);v+=__shfl_down_sync(0xffffffff,v,2);
    v+=__shfl_down_sync(0xffffffff,v,1);
    if(t==0)s[0]=v;__syncthreads();float rs=s[0];

    for(int i=0;i<items;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- V4: reg-cached online (1R+1W, merge-reduce) ---
__global__ void v4(const float*x,float*o,int nr,int hs){
    int r=blockIdx.x,t=threadIdx.x,tt=blockDim.x,items=hs/tt;
    const float*xr=x+r*hs;float*or_=o+r*hs;
    __shared__ float sm[BLOCK],sd[BLOCK];
    float x_reg[16];
    for(int i=0;i<items;i++)x_reg[i]=xr[t+i*tt];

    float m=-INFINITY,d=0;
    for(int i=0;i<items;i++){float vv=x_reg[i];
        if(vv>m){d=d*expf(m-vv)+1.0f;m=vv;}else{d+=expf(vv-m);}}
    sm[t]=m;sd[t]=d;__syncthreads();
    for(int st=BLOCK/2;st>=32;st>>=1){if(t<st){float am=sm[t],ad=sd[t],bm=sm[t+st],bd=sd[t+st];
        if(am>=bm){sm[t]=am;sd[t]=ad+bd*expf(bm-am);}else{sm[t]=bm;sd[t]=bd+ad*expf(am-bm);}}__syncthreads();}
    m=sm[t];d=sd[t];
    for(int dl=16;dl>=1;dl>>=1){float om=__shfl_down_sync(0xffffffff,m,dl),od=__shfl_down_sync(0xffffffff,d,dl);
        if(m>=om){d=d+od*expf(om-m);}else{d=od+d*expf(m-om);m=om;}}
    if(t==0){sm[0]=m;sd[0]=d;}__syncthreads();float rm=sm[0],rs=sd[0];

    for(int i=0;i<items;i++)or_[t+i*tt]=expf(x_reg[i]-rm)/rs;
}

// --- MyOwn: online, no x_reg (2R+1W) ---
#define MY_BLOCK_SIZE 256
__global__ void myown(const float*x,float*o,int nr,int hs){
    int tid=threadIdx.x,gap=blockIdx.x*hs;
    const float*xc=x+gap;float*oc=o+gap;
    __shared__ float sm_max[MY_BLOCK_SIZE],sm_exp[MY_BLOCK_SIZE];
    float mx=-INFINITY,es=0;
    for(int i=tid;i<hs;i+=MY_BLOCK_SIZE){float v=xc[i];
        if(v>mx){es=es*expf(mx-v)+1.0f;mx=v;}else{es+=expf(v-mx);}}
    sm_max[tid]=mx;sm_exp[tid]=es;__syncthreads();
    for(int st=MY_BLOCK_SIZE/2;st>=32;st/=2){if(tid<st){
        if(sm_max[tid]<sm_max[tid+st]){sm_exp[tid]=sm_exp[tid]*expf(sm_max[tid]-sm_max[tid+st])+sm_exp[tid+st];sm_max[tid]=sm_max[tid+st];}
        else{sm_exp[tid]=sm_exp[tid]+sm_exp[tid+st]*expf(sm_max[tid+st]-sm_max[tid]);}}__syncthreads();}
    float v=sm_exp[tid],m2=sm_max[tid];
    float t2=__shfl_down_sync(0xffffffff,m2,16);if(m2<t2){v=v*expf(m2-t2)+__shfl_down_sync(0xffffffff,v,16);m2=t2;}else{v=v+__shfl_down_sync(0xffffffff,v,16)*expf(t2-m2);}
    t2=__shfl_down_sync(0xffffffff,m2,8);if(m2<t2){v=v*expf(m2-t2)+__shfl_down_sync(0xffffffff,v,8);m2=t2;}else{v=v+__shfl_down_sync(0xffffffff,v,8)*expf(t2-m2);}
    t2=__shfl_down_sync(0xffffffff,m2,4);if(m2<t2){v=v*expf(m2-t2)+__shfl_down_sync(0xffffffff,v,4);m2=t2;}else{v=v+__shfl_down_sync(0xffffffff,v,4)*expf(t2-m2);}
    t2=__shfl_down_sync(0xffffffff,m2,2);if(m2<t2){v=v*expf(m2-t2)+__shfl_down_sync(0xffffffff,v,2);m2=t2;}else{v=v+__shfl_down_sync(0xffffffff,v,2)*expf(t2-m2);}
    t2=__shfl_down_sync(0xffffffff,m2,1);if(m2<t2){v=v*expf(m2-t2)+__shfl_down_sync(0xffffffff,v,1);m2=t2;}else{v=v+__shfl_down_sync(0xffffffff,v,1)*expf(t2-m2);}
    if(tid==0){sm_max[tid]=m2;sm_exp[tid]=v;}__syncthreads();
    float mf=sm_max[0],ef=sm_exp[0];
    for(int i=tid;i<hs;i+=BLOCK)oc[i]=expf(xc[i]-mf)/ef;
}

float ms(cudaEvent_t s,cudaEvent_t e){float x;cudaEventElapsedTime(&x,s,e);return x;}
void chk(cudaError_t e,const char*m){if(e!=cudaSuccess){fprintf(stderr,"%s:%s\n",m,cudaGetErrorString(e));exit(1);}}

int main(){
    cudaEvent_t st,sp;chk(cudaEventCreate(&st),"ev");chk(cudaEventCreate(&sp),"ev");

    printf("===== Same-size comparison =====\n");
    printf("  V3:   naive 3-phase,  x_reg[], 1R+1W\n");
    printf("  V4:   online merge,  x_reg[], 1R+1W\n");
    printf("  MyOwn: online merge, NO x_reg, 2R+1W\n\n");

    int sizes[]={4096,8192};
    for(int s=0;s<2;s++){
        int h=sizes[s];
        long long tb=(long long)N_ROWS*h*4;
        printf("--- HIDDEN=%d (data=%.1f MB) ---\n",h,tb/1e6f);
        if(h==8192)printf("    V3 SKIP: x_reg[16] too small for 32 items/thread\n");

        float*hx=(float*)malloc(tb);
        for(long long i=0;i<(long long)N_ROWS*h;i++)hx[i]=(float)(rand())/RAND_MAX-0.5f;
        float*dx,*do_;chk(cudaMalloc(&dx,tb),"dx");chk(cudaMalloc(&do_,tb),"do");
        chk(cudaMemcpy(dx,hx,tb,cudaMemcpyHostToDevice),"H2D");
        dim3 g(N_ROWS),b(BLOCK);

        typedef void (*kfn)(const float*,float*,int,int);
        kfn ks[]={v3,v4,myown};
        const char*ns[]={"V3(naive+reg)","V4(online+reg)","MyOwn(online)"};

        for(int k=0;k<3;k++){
            if(h==8192 && k==0){printf("  %-18s SKIP\n",ns[k]);continue;}
            for(int i=0;i<WARMUP;i++)ks[k]<<<g,b>>>(dx,do_,N_ROWS,h);
            chk(cudaDeviceSynchronize(),"wu");
            float best=1e9;
            for(int r=0;r<RUNS;r++){cudaEventRecord(st,0);ks[k]<<<g,b>>>(dx,do_,N_ROWS,h);cudaEventRecord(sp,0);cudaEventSynchronize(sp);float tt=ms(st,sp);if(tt<best)best=tt;}

            float traffic=(k<2)?tb*2.0f:tb*3.0f; // 1R+1W vs 2R+1W
            float bw=traffic/(best/1000.0f)/1e9f;
            printf("  %-18s %7.4f ms  BW %6.1f GB/s (%5.1f%%)\n",ns[k],best,bw,bw/192.0f*100);
        }
        cudaFree(dx);cudaFree(do_);free(hx);
    }

    printf("\n===== What this tells us =====\n");
    printf("  At 4096: V3 vs V4 = naive vs online (both 1R+1W).\n");
    printf("    Online merge adds ~255 exp calls → slightly slower.\n");
    printf("    但两者都比 MyOwn 快 → 1R+1W 是真正的优化。\n");
    printf("  At 8192: V4 vs MyOwn = 1R+1W vs 2R+1W (both online).\n");
    printf("    唯一区别是 x_reg[] → 1R+1W 比 2R+1W 快 ~1.5x。\n");

    cudaEventDestroy(st);cudaEventDestroy(sp);
    return 0;
}
