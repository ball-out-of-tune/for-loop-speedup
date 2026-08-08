#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d\n",e);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

// V9 GEMM (128x128 tile, 4 warps, cp.async double-buffered)
__global__ void gemm_v9(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*128+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half Ab[2][128][16],Bb[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);
    for(int idx=threadIdx.x;idx<128*16/2;idx+=128){int pos=idx*2,r=pos/16,col=pos%16,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<16*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=16;kb<K;kb+=16){int wb=1-rb;
        for(int c=threadIdx.x;c<128*16/8;c+=128){int p=c*8,r=p/16,col=p%16,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<16*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// Scale + softmax: S[rows][cols] in FP32 -> P[rows][cols] in FP16
__global__ void scale_softmax(int rows, int cols, const float* S, half* P, float sc) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows)return;
    float mx=-1e9f;
    for(int j=0;j<cols;j++){float v=S[r*cols+j]*sc;if(v>mx)mx=v;}
    float sm=0;
    for(int j=0;j<cols;j++)sm+=expf(S[r*cols+j]*sc-mx);
    float inv=1.0f/sm;
    for(int j=0;j<cols;j++)P[r*cols+j]=__float2half(expf(S[r*cols+j]*sc-mx)*inv);
}

void cpu_attn(int N,int d,const float* Q,const float* K,const float* V,float* O,float scale){
    float* S=(float*)malloc(N*N*4);
    for(int i=0;i<N;i++)for(int j=0;j<N;j++){float s=0;for(int k=0;k<d;k++)s+=Q[i*d+k]*K[j*d+k];S[i*N+j]=s*scale;}
    for(int i=0;i<N;i++){float mx=-1e9f,sm=0;for(int j=0;j<N;j++)mx=fmaxf(mx,S[i*N+j]);
        for(int j=0;j<N;j++){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}
        for(int j=0;j<N;j++)S[i*N+j]/=sm;}
    for(int i=0;i<N;i++)for(int k=0;k<d;k++){float s=0;for(int j=0;j<N;j++)s+=S[i*N+j]*V[j*d+k];O[i*d+k]=s;}
    free(S);
}

int main(){
    int N=2048,D=64;float scale=1.0f/sqrtf(64.0f);
    int Np=((N+127)/128)*128;
    printf("GEMM Attention test N=%d D=%d N_pad=%d\n",N,D,Np);

    float*hQ=(float*)calloc(Np*D,4),*hK=(float*)calloc(Np*D,4),*hV=(float*)calloc(Np*D,4);
    float*hO_cpu=(float*)calloc(N*D,4),*hO_gpu=(float*)calloc(N*D,4);
    srand(42);
    for(int i=0;i<N*D;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}
    cpu_attn(N,D,hQ,hK,hV,hO_cpu,scale);

    float *dQ32,*dK32,*dV32,*dS,*dO;half *dQ16,*dK16,*dV16,*dP;
    CHK(cudaMalloc(&dQ32,Np*D*4));CHK(cudaMalloc(&dK32,Np*D*4));CHK(cudaMalloc(&dV32,Np*D*4));
    CHK(cudaMalloc(&dQ16,Np*D*2));CHK(cudaMalloc(&dK16,Np*D*2));CHK(cudaMalloc(&dV16,Np*D*2));
    CHK(cudaMalloc(&dS,Np*Np*4));CHK(cudaMalloc(&dP,Np*Np*2));CHK(cudaMalloc(&dO,Np*D*4));
    CHK(cudaMemset(dQ32,0,Np*D*4));CHK(cudaMemset(dK32,0,Np*D*4));CHK(cudaMemset(dV32,0,Np*D*4));
    CHK(cudaMemcpy(dQ32,hQ,N*D*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dK32,hK,N*D*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dV32,hV,N*D*4,cudaMemcpyHostToDevice));
    f32tof16<<<(Np*D+255)/256,256>>>(Np*D,dQ32,dQ16);
    f32tof16<<<(Np*D+255)/256,256>>>(Np*D,dK32,dK16);
    f32tof16<<<(Np*D+255)/256,256>>>(Np*D,dV32,dV16);
    CHK(cudaDeviceSynchronize());

    dim3 blk(128),g1((Np+127)/128,(Np+127)/128),g2(((D+127)/128),(Np+127)/128);

    // Warmup
    for(int i=0;i<10;i++){gemm_v9<<<g1,blk>>>(Np,Np,D,dQ16,dK16,dS);scale_softmax<<<(N+255)/256,256>>>(N,Np,dS,dP,scale);gemm_v9<<<g2,blk>>>(N,D,Np,dP,dV16,dO);}
    CHK(cudaDeviceSynchronize());
    if(cudaGetLastError()!=cudaSuccess){printf("WARMUP FAILED E%d\n",cudaGetLastError());return 1;}

    int reps=50;cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<reps;i++){gemm_v9<<<g1,blk>>>(Np,Np,D,dQ16,dK16,dS);scale_softmax<<<(N+255)/256,256>>>(N,Np,dS,dP,scale);gemm_v9<<<g2,blk>>>(N,D,Np,dP,dV16,dO);}
    cudaEventRecord(e);cudaEventSynchronize(e);
    float ms;cudaEventElapsedTime(&ms,s,e);

    CHK(cudaMemcpy(hO_gpu,dO,N*D*4,cudaMemcpyDeviceToHost));
    float mx=0;for(int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_cpu[i]);if(d>mx)mx=d;}
    double flops=4.0*N*N*D,tf=flops/(ms/reps/1000.0)/1e12;
    printf("N=%d: %.3fms %.2f TFLOPS MaxErr=%.4f\n",N,ms/reps,tf,mx);

    free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
    cudaFree(dQ32);cudaFree(dK32);cudaFree(dV32);cudaFree(dQ16);cudaFree(dK16);cudaFree(dV16);
    cudaFree(dS);cudaFree(dP);cudaFree(dO);
    return 0;
}
