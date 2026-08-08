#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>
using namespace std;

// ============================================================
// 工具: 构建二维行指针 — A/B 用 const float**, C 用 float**
// ============================================================

// 构建只读行指针: 从平铺数据 -> const float**
void build_const_rows(const vector<float>& flat, int rows, int cols,
                      vector<const float*>& out) {
    out.resize(rows);
    for (int i = 0; i < rows; i++)
        out[i] = &flat[i * cols];
}

// 构建可写行指针: 从平铺数据 -> float**
void build_rows(const vector<float>& flat, int rows, int cols,
                vector<float*>& out) {
    out.resize(rows);
    for (int i = 0; i < rows; i++)
        out[i] = const_cast<float*>(&flat[i * cols]);
}

void random_fill(vector<float>& v) {
    for (auto& x : v)
        x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

float max_error(const vector<float>& a, const vector<float>& b) {
    float e = 0;
    for (size_t i = 0; i < a.size(); i++)
        e = max(e, fabsf(a[i] - b[i]));
    return e;
}

void print(const vector<float>& flat, int M, int K, const char* name,
           const vector<float*>& rows) {
    cout << name << " (" << M << "x" << K << "):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            printf("%7.2f", rows[i][j]);
        cout << "\n";
    }
    cout << "\n";
}


// ============================================================
// 1. 你的版本 (保留: 块编号风格, 已修正)
// ============================================================
void gemm_my_original(int M, int N, int K,
                      const float** A, const float** B, float** C,
                      int block_size) {
    for (int i = 0; i < M / block_size; i++) {               // C 行块
        for (int j = 0; j < K / block_size; j++) {           // C 列块
            for (int k = 0; k < N / block_size; k++) {       // 归约维块

                for (int inner_i = i * block_size;
                     inner_i < (i + 1) * block_size; inner_i++) {
                    for (int inner_j = j * block_size;
                         inner_j < (j + 1) * block_size; inner_j++) {
                        for (int inner_k = 0;
                             inner_k < block_size; inner_k++) {
                            C[inner_i][inner_j] +=
                                A[inner_i][k * block_size + inner_k] *
                                B[k * block_size + inner_k][inner_j];
                        }
                    }
                }

            }
        }
    }
}


// ============================================================
// 2. 朴素参考实现 — 保证正确
// ============================================================
void gemm_naive(int M, int N, int K,
                const float** A, const float** B, float** C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0;
            for (int p = 0; p < N; p++)
                sum += A[i][p] * B[p][j];
            C[i][j] = sum;
        }
    }
}


// ============================================================
// 3. 分块 GEMM (偏移量风格, 带边界保护)
// ============================================================
void gemm_tiled(int M, int N, int K,
                const float** A, const float** B, float** C,
                int block_size) {
    for (int i = 0; i < M; i += block_size) {
        for (int j = 0; j < K; j += block_size) {
            for (int k = 0; k < N; k += block_size) {

                int i_end = min(i + block_size, M);
                int j_end = min(j + block_size, K);
                int k_end = min(k + block_size, N);

                for (int ii = i; ii < i_end; ii++) {
                    for (int jj = j; jj < j_end; jj++) {
                        float sum = 0;
                        for (int kk = k; kk < k_end; kk++)
                            sum += A[ii][kk] * B[kk][jj];
                        C[ii][jj] += sum;
                    }
                }

            }
        }
    }
}


// ============================================================
// 测试入口
// ============================================================
int main() {
    srand(42);

    // ====== 小规模: 可打印对比 ======
    {
        int M = 8, N = 6, K = 8, bs = 2;

        // 数据存储
        vector<float> A_flat(M * N), B_flat(N * K);
        vector<float> C_naive_flat(M * K), C_my_flat(M * K), C_tiled_flat(M * K);
        random_fill(A_flat);
        random_fill(B_flat);

        // 行指针
        vector<const float*> A_r, B_r;          // 只读
        vector<float*>       C_n_r, C_m_r, C_t_r; // 可写
        build_const_rows(A_flat, M, N, A_r);
        build_const_rows(B_flat, N, K, B_r);
        build_rows(C_naive_flat, M, K, C_n_r);
        build_rows(C_my_flat,    M, K, C_m_r);
        build_rows(C_tiled_flat, M, K, C_t_r);

        // 计算
        gemm_naive      (M, N, K, A_r.data(), B_r.data(), C_n_r.data());
        gemm_my_original(M, N, K, A_r.data(), B_r.data(), C_m_r.data(), bs);
        gemm_tiled      (M, N, K, A_r.data(), B_r.data(), C_t_r.data(), bs);

        print(A_flat, M, N, "A", *reinterpret_cast<vector<float*>*>(&A_r));
        print(B_flat, N, K, "B", *reinterpret_cast<vector<float*>*>(&B_r));
        print(C_naive_flat, M, K, "C_naive (ground truth)", C_n_r);
        print(C_my_flat,    M, K, "C_my_original", C_m_r);
        print(C_tiled_flat, M, K, "C_tiled", C_t_r);

        printf("my_original vs naive: max_err = %.2e  %s\n",
               max_error(C_naive_flat, C_my_flat),
               max_error(C_naive_flat, C_my_flat) < 1e-5f ? "PASS" : "FAIL");
        printf("tiled       vs naive: max_err = %.2e  %s\n\n",
               max_error(C_naive_flat, C_tiled_flat),
               max_error(C_naive_flat, C_tiled_flat) < 1e-5f ? "PASS" : "FAIL");
    }

    // ====== 中规模: 误差验证 ======
    {
        int M = 256, N = 128, K = 256, bs = 32;

        vector<float> A_flat(M * N), B_flat(N * K);
        vector<float> C_naive_flat(M * K), C_my_flat(M * K), C_tiled_flat(M * K);
        random_fill(A_flat);
        random_fill(B_flat);

        vector<const float*> A_r, B_r;
        vector<float*> C_n_r, C_m_r, C_t_r;
        build_const_rows(A_flat, M, N, A_r);
        build_const_rows(B_flat, N, K, B_r);
        build_rows(C_naive_flat, M, K, C_n_r);
        build_rows(C_my_flat,    M, K, C_m_r);
        build_rows(C_tiled_flat, M, K, C_t_r);

        gemm_naive      (M, N, K, A_r.data(), B_r.data(), C_n_r.data());
        gemm_my_original(M, N, K, A_r.data(), B_r.data(), C_m_r.data(), bs);
        gemm_tiled      (M, N, K, A_r.data(), B_r.data(), C_t_r.data(), bs);

        printf("Medium (M=%d,N=%d,K=%d,bs=%d):\n", M, N, K, bs);
        printf("  my_original vs naive: max_err = %.2e  %s\n",
               max_error(C_naive_flat, C_my_flat),
               max_error(C_naive_flat, C_my_flat) < 1e-5f ? "PASS" : "FAIL");
        printf("  tiled       vs naive: max_err = %.2e  %s\n\n",
               max_error(C_naive_flat, C_tiled_flat),
               max_error(C_naive_flat, C_tiled_flat) < 1e-5f ? "PASS" : "FAIL");
    }

    // ====== 大规模: 性能 ======
    {
        int M = 1024, N = 512, K = 1024, bs = 32;

        vector<float> A_flat(M * N), B_flat(N * K), C_flat(M * K);
        random_fill(A_flat);
        random_fill(B_flat);

        vector<const float*> A_r, B_r;
        vector<float*> C_r;
        build_const_rows(A_flat, M, N, A_r);
        build_const_rows(B_flat, N, K, B_r);
        build_rows(C_flat, M, K, C_r);

        auto t0 = chrono::high_resolution_clock::now();
        gemm_tiled(M, N, K, A_r.data(), B_r.data(), C_r.data(), bs);
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();
        double gflops = 2.0 * M * N * K / sec / 1e9;

        printf("Large (M=%d,N=%d,K=%d,bs=%d):\n", M, N, K, bs);
        printf("  tiled: %.0f ms  (%.1f GFLOPS)\n", sec * 1000, gflops);

        // 两次运行一致性检查
        vector<float> C2_flat(M * K);
        vector<float*> C2_r;
        build_rows(C2_flat, M, K, C2_r);
        gemm_tiled(M, N, K, A_r.data(), B_r.data(), C2_r.data(), bs);
        printf("  deterministic check: max_err = %.2e  %s\n",
               max_error(C_flat, C2_flat),
               max_error(C_flat, C2_flat) < 1e-10f ? "PASS" : "FAIL");
    }

    cout << "\nDone.\n";
    return 0;
}
