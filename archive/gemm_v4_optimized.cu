/**
 * V4: Aggressively optimized GEMM for RTX 5090 (sm_120)
 * Key improvements:
 * - 256 threads (8 warps) → better occupancy
 * - uint4 vectorized prefetch (16-byte) → faster loads
 * - 256x128 tile → more data reuse
 * - Persistent kernel for small M → eliminates launch overhead
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
using namespace nvcuda;

double now_ms(){struct timeval tv;gettimeofday(&tv,NULL);return tv.tv_sec*1000.0+tv.tv_usec/1000.0;}
#define CK(call) do{cudaError_t e=call;if(e!=cudaSuccess){fprintf(stderr,"ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CBLAS(call) do{cublasStatus_t e=call;if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CBLAS err %d\n",e);exit(1);}}while(0)

// ============ Kernel 1: 256-thread, 256x128 tile ============
__global__ void gemm_256t(const half* __restrict__ A,const half* __restrict__ B,
                           float* __restrict__ C,int M,int N,int K){
    // 8 warps in 4(row) x 2(col) layout
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*256+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half As[2][256][16],Bs[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    #pragma unroll
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    // Prefetch K-tile 0 with uint4 (16-byte aligned, 8x half per load)
    for(int idx=threadIdx.x;idx<256*16/8;idx+=256){
        int pos=idx*8,r=pos/16,c8=pos%16;
        int gr=blockIdx.y*256+r,gc=c8;
        if(gr<M&&gc+7<K)*(uint4*)&As[0][r][c8]=*(const uint4*)&A[gr*K+gc];
    }
    for(int idx=threadIdx.x;idx<16*128/8;idx+=256){
        int pos=idx*8,r=pos/128,c8=pos%128;
        int gr=r,gc=blockIdx.x*128+c8;
        if(gr<K&&gc+7<N)*(uint4*)&Bs[0][r][c8]=*(const uint4*)&B[gr*N+gc];
    }
    __syncthreads();
    int rb=0;
    bool use_cp=((unsigned long long)K&7)==0;

    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        // cp.async next tile
        for(int ch=threadIdx.x;ch<256*16/8;ch+=256){
            int pos=ch*8,r=pos/16,c8=pos%16;
            int gr=blockIdx.y*256+r,gc=kb+c8;
            if(gr<M&&gc+7<K){
                unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
                if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
                else for(int j=0;j<8&&gc+j<K;j++)As[wb][r][c8+j]=A[gr*K+gc+j];
            }
        }
        for(int ch=threadIdx.x;ch<16*128/8;ch+=256){
            int pos=ch*8,r=pos/128,c8=pos%128;
            int gr=kb+r,gc=blockIdx.x*128+c8;
            if(gr<K&&gc+7<N){
                unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
                if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
                else for(int j=0;j<8&&gc+j<N;j++)Bs[wb][r][c8+j]=B[gr*N+gc+j];
            }
        }
        if(use_cp)asm volatile("cp.async.commit_group;\n"::);
        else __syncthreads();

        // WMMA compute from read_buf
        #pragma unroll
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);

        if(use_cp)asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        rb=wb;
    }
    // Last tile
    #pragma unroll
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
    #pragma unroll
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    // Store
    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++){
        int fr=r0+mi*16,fc=c0+ni*16;
        if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

// ============ Kernel 2: Optimized for small K (flat matrices) ============
// For K <= 256: load entire K dimension into shared memory at once
__global__ void gemm_smallK(const half* __restrict__ A,const half* __restrict__ B,
                             float* __restrict__ C,int M,int N,int K){
    int tid=threadIdx.x;
    int r0=blockIdx.y*128,c0=blockIdx.x*128;

    // Load entire A tile (128xK) and B tile (Kx128) into shared mem
    // Use dynamic shared memory
    extern __shared__ half smem[];
    half* As=smem;               // [128][K]
    half* Bs=&smem[128*K];       // [K][128]

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    #pragma unroll
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    // Cooperative load: all threads load A and B
    int total_a=128*K,total_b=K*128;
    for(int i=tid;i<total_a;i+=256){
        int r=i/K,c2=i%K;
        As[r*K+c2]=(blockIdx.y*128+r<M&&c2<K)?A[(blockIdx.y*128+r)*K+c2]:__float2half(0.0f);
    }
    for(int i=tid;i<total_b;i+=256){
        int r=i/128,c2=i%128;
        Bs[r*128+c2]=(r<K&&blockIdx.x*128+c2<N)?B[r*N+blockIdx.x*128+c2]:__float2half(0.0f);
    }
    __syncthreads();

    // All K tiles in shared memory - no K-loop!
    int wy=tid/64,wx=(tid%64)/32; // simplified: thread to warp mapping
    if(wy>=2)wy=wy%2;
    if(wx>=2)wx=wx%2;

    for(int kb=0;kb<K;kb+=16){
        #pragma unroll
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[(wy*64+mi*16)*K+kb],K);
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[kb*128+wx*64+ni*16],128);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    }
    // Store
    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++){
        int fr=r0+wy*64+mi*16,fc=c0+wx*64+ni*16;
        if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

// ============ Kernel 3: Persistent kernel for LLM decode (small M) ============
// One block processes multiple tiles to eliminate launch overhead
__global__ void gemm_persistent(const half* __restrict__ A,const half* __restrict__ B,
                                 float* __restrict__ C,int M,int N,int K,
                                 int total_tiles){
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    __shared__ half As[2][128][16],Bs[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];

    // Each block handles multiple tiles from a work queue
    int tiles_per_block=(total_tiles+gridDim.x-1)/gridDim.x;
    int start_tile=blockIdx.x*tiles_per_block;
    int end_tile=min(start_tile+tiles_per_block,total_tiles);

    for(int tile=start_tile;tile<end_tile;tile++){
        int tile_row=(tile/((N+127)/128))*128;
        int tile_col=(tile%((N+127)/128))*128;
        int r0=tile_row+wy*64,c0=tile_col+wx*64;

        #pragma unroll
        for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

        // Prefetch
        for(int idx=threadIdx.x;idx<128*16;idx+=128){
            int r=idx/16,c2=idx%16;
            int gr=tile_row+r,gc=c2;
            As[0][r][c2]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.0f);
        }
        for(int idx=threadIdx.x;idx<16*128;idx+=128){
            int r=idx/128,c2=idx%128;
            int gr=r,gc=tile_col+c2;
            Bs[0][r][c2]=(gr<K&&gc<N)?B[gr*N+gc]:__float2half(0.0f);
        }
        __syncthreads();
        int rb=0;
        bool use_cp=((unsigned long long)K&7)==0;

        for(int kb=16;kb<K;kb+=16){
            int wb=1-rb;
            for(int ch=threadIdx.x;ch<128*16/8;ch+=128){
                int pos=ch*8,r=pos/16,c8=pos%16;
                int gr=tile_row+r,gc=kb+c8;
                if(gr<M&&gc+7<K){
                    unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
                    if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
                    else for(int j=0;j<8&&gc+j<K;j++)As[wb][r][c8+j]=A[gr*K+gc+j];
                }
            }
            for(int ch=threadIdx.x;ch<16*128/8;ch+=128){
                int pos=ch*8,r=pos/128,c8=pos%128;
                int gr=kb+r,gc=tile_col+c8;
                if(gr<K&&gc+7<N){
                    unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
                    if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
                    else for(int j=0;j<8&&gc+j<N;j++)Bs[wb][r][c8+j]=B[gr*N+gc+j];
                }
            }
            if(use_cp)asm volatile("cp.async.commit_group;\n"::);
            else __syncthreads();

            #pragma unroll
            for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
            #pragma unroll
            for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
            #pragma unroll
            for(int mi=0;mi<4;mi++)
            #pragma unroll
            for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);

            if(use_cp)asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
            rb=wb;
        }
        #pragma unroll
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++){
            int fr=r0+mi*16,fc=c0+ni*16;
            if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
        }
    }
}


// ============ Auto-dispatch ============
void run_gemm(const half* dA,const half* dB,float* dC,int M,int N,int K,cudaStream_t s=0){
    if(K<=256 && M>=128 && N>=128){
        // Small K: use smallK kernel (load all K into shared memory)
        int shmem=128*K*2+K*128*2; // bytes for As[128][K] + Bs[K][128]
        dim3 blk(256),grd((N+127)/128,(M+127)/128);
        gemm_smallK<<<grd,blk,shmem,s>>>(dA,dB,dC,M,N,K);
    }else if(M<=128 && (long long)M*N>1024*1024){
        // Small M (LLM decode): use persistent kernel
        int tiles_M=(M+127)/128,tiles_N=(N+127)/128;
        int total_tiles=tiles_M*tiles_N;
        int blocks=min(total_tiles,170*8); // 170 SMs * 8 blocks each
        gemm_persistent<<<blocks,128,0,s>>>(dA,dB,dC,M,N,K,total_tiles);
    }else if(M>=512){
        // Large M: 256-thread kernel
        dim3 blk(256),grd((N+127)/128,(M+255)/256);
        gemm_256t<<<grd,blk,0,s>>>(dA,dB,dC,M,N,K);
    }else{
        // Default
        dim3 blk(128),grd((N+127)/128,(M+127)/128);
        gemm_256t<<<grd,blk,0,s>>>(dA,dB,dC,M,N,K);
    }
}


// ============ Main benchmark ============
int main(){
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    printf("GPU: %s | CC %d.%d | SMs %d\n\n",p.name,p.major,p.minor,p.multiProcessorCount);
    cublasHandle_t h;CBLAS(cublasCreate(&h));

    struct{int M,N,K;const char* t;}T[]={
        {512,512,512,"sq-512"},{1024,1024,1024,"sq-1K"},
        {2048,2048,2048,"sq-2K"},{4096,4096,4096,"sq-4K"},
        {256,256,8192,"lK-8K"},{512,512,16384,"lK-16K"},{1024,1024,32768,"lK-32K"},
        {1,4096,4096,"d-1"},{1,8192,8192,"d-L"},{4,4096,4096,"d-4"},
        {8,4096,4096,"d-8"},{16,4096,4096,"d-16"},{32,4096,4096,"d-32"},
        {64,4096,4096,"d-64"},{128,4096,4096,"d-128"},
        {512,4096,4096,"p-512"},{1024,4096,4096,"p-1K"},{2048,4096,4096,"p-2K"},
        {4096,4096,64,"f-64"},{8192,8192,128,"f-128"},
        {1000,2000,500,"unalign"},
        {1,12288,12288,"gpt3-d"},{128,12288,12288,"gpt3-p"},
        {1,8192,28672,"llama-d"},{128,8192,28672,"llama-p"},
        {1,1024,3072,"qwen3-d"},{128,1024,3072,"qwen3-p"},
        {0,0,0,NULL}};
    int n=0;while(T[n].M)n++;

    printf("V4: 256-thread + uint4 prefetch + persistent kernel\n");
    printf("%-20s %-10s %10s %10s %8s %10s %10s\n","Shape","Category","Custom ms","cuBLAS ms","Speedup","CusTF","cuBLASTF");
    printf("%.20s %-10s %10s %10s %8s %10s %10s\n","--------------------","----------","----------","----------","--------","----------","----------");

    double totC=0,totB=0;int wins=0;
    for(int i=0;i<n;i++){
        int M=T[i].M,Nn=T[i].N,K=T[i].K;
        half *dA,*dB,*hA=(half*)malloc(M*K*2),*hB=(half*)malloc(K*Nn*2);
        float *dCc,*dCb;
        for(int j=0;j<M*K;j++)hA[j]=__float2half(((float)rand()/RAND_MAX-0.5f)*0.1f);
        for(int j=0;j<K*Nn;j++)hB[j]=__float2half(((float)rand()/RAND_MAX-0.5f)*0.1f);
        CK(cudaMalloc(&dA,M*K*2));CK(cudaMemcpy(dA,hA,M*K*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dB,K*Nn*2));CK(cudaMemcpy(dB,hB,K*Nn*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dCc,M*Nn*4));CK(cudaMalloc(&dCb,M*Nn*4));
        free(hA);free(hB);

        int wup=3,itr=20;
        long long fl=2LL*M*Nn*K;
        if(fl>1e8){wup=2;itr=10;}if(fl>5e8){wup=1;itr=5;}

        // Custom kernel (auto-dispatched)
        CK(cudaGetLastError());
        for(int j=0;j<wup;j++)run_gemm(dA,dB,dCc,M,Nn,K);
        CK(cudaDeviceSynchronize());
        double t0=now_ms();
        for(int j=0;j<itr;j++)run_gemm(dA,dB,dCc,M,Nn,K);
        CK(cudaDeviceSynchronize());
        double ct=(now_ms()-t0)/itr;

        // cuBLAS
        float al=1,be=0;
        for(int j=0;j<wup;j++)
            CBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Nn,M,K,&al,dB,CUDA_R_16F,Nn,dA,CUDA_R_16F,K,&be,dCb,CUDA_R_32F,Nn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        t0=now_ms();
        for(int j=0;j<itr;j++)
            CBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Nn,M,K,&al,dB,CUDA_R_16F,Nn,dA,CUDA_R_16F,K,&be,dCb,CUDA_R_32F,Nn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        double bt=(now_ms()-t0)/itr;

        double sp=bt/ct,tf_c=(2.0*M*Nn*K)/(ct/1000.0)/1e12,tf_b=(2.0*M*Nn*K)/(bt/1000.0)/1e12;
        totC+=ct;totB+=bt;if(sp>=1.0)wins++;
        char sh[32];snprintf(sh,sizeof(sh),"%dx%dx%d",M,Nn,K);
        printf("%-20s %-10s %10.4f %10.4f %7.2fx %10.1f %10.1f\n",sh,T[i].t,ct,bt,sp,tf_c,tf_b);
        CK(cudaFree(dA));CK(cudaFree(dB));CK(cudaFree(dCc));CK(cudaFree(dCb));
    }
    printf("%.20s %-10s %10s %10s %8s %10s %10s\n","--------------------","----------","----------","----------","--------","----------","----------");
    printf("%-20s %-10s %10.2f %10.2f %7.2fx\n","TOTAL","",totC,totB,totB/totC);
    printf("Wins: %d/%d\n",wins,n);
    cublasDestroy(h);
    return 0;
}
