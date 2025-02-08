#define min(a, b) ((a) < (b) ? (a) : (b))

void macro_kernel(int M, int N, int K, double *restrict A, int LDA, double *restrict B, int LDB, double *restrict C, int LDC)
{
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            double bkn = B[k + n * LDB];
            for (int m = 0; m < M; m++) {
                C[m + n * LDC] += A[m + k * LDA] * bkn;
            }
        }
    }
}

#define M_BLOCK_SIZE 256
#define N_BLOCK_SIZE 256
#define K_BLOCK_SIZE 256

void dgemm(int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    int LDA = M;
    int LDB = K;
    int LDC = M;

    int m_inc, n_inc, k_inc;
    for (int k = 0; k < K;) {
        k_inc = min(K_BLOCK_SIZE, K - k);

        for (int n = 0; n < N;) {
            n_inc = min(N_BLOCK_SIZE, N - n);

            for (int m = 0; m < M;) {
                m_inc = min(M_BLOCK_SIZE, M - m);
                macro_kernel(
                    m_inc, n_inc, k_inc,
                    A + m + k * LDA, LDA,
                    B + k + n * LDB, LDB,
                    C + m + n * LDC, LDC
                );
                m += m_inc;
            }
            n += n_inc;
        }
        k += k_inc;
    }
}
