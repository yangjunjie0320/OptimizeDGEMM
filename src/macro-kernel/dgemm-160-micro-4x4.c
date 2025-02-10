#define min(a, b) ((a) < (b) ? (a) : (b))
void edge_block(int M, int N, int K, double *restrict A, int LDA, double *restrict B, int LDB, double *restrict C, int LDC)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double cmn = C[m + n * LDC];
            for (int k = 0; k < K; k++) {
                double amk = A[m + k * LDA];
                double bkn = B[k + n * LDB];
                cmn += amk * bkn;
            }
            C[m + n * LDC] = cmn;
        }
    }
}

#define BLOCK_SIZE 4
void micro_kernel(int M, int N, int K, double *restrict A, int LDA, double *restrict B, int LDB, double *restrict C, int LDC)
{
    int M0 = M, N0 = N, K0 = K;
    int M4 = M0 & -BLOCK_SIZE;
    int N4 = N0 & -BLOCK_SIZE;
    int K4 = K0 & -BLOCK_SIZE;

    register double c00, c01, c10, c11;
    register double c20, c21, c30, c31;
    register double c02, c03, c12, c13;
    register double c22, c23, c32, c33;

    for (int m = 0; m < M4; m += BLOCK_SIZE) {
        for (int n = 0; n < N4; n += BLOCK_SIZE) {
            c00 = C[(m + 0) + (n + 0) * LDC];
            c01 = C[(m + 0) + (n + 1) * LDC];
            c10 = C[(m + 1) + (n + 0) * LDC];
            c11 = C[(m + 1) + (n + 1) * LDC];
            c20 = C[(m + 2) + (n + 0) * LDC];
            c21 = C[(m + 2) + (n + 1) * LDC];
            c30 = C[(m + 3) + (n + 0) * LDC];
            c31 = C[(m + 3) + (n + 1) * LDC];
            c02 = C[(m + 0) + (n + 2) * LDC];
            c03 = C[(m + 0) + (n + 3) * LDC];
            c12 = C[(m + 1) + (n + 2) * LDC];
            c13 = C[(m + 1) + (n + 3) * LDC];
            c22 = C[(m + 2) + (n + 2) * LDC];
            c23 = C[(m + 2) + (n + 3) * LDC];
            c32 = C[(m + 3) + (n + 2) * LDC];
            c33 = C[(m + 3) + (n + 3) * LDC];

            for (int k = 0; k < K4; k++) {
                register double a0k = A[(m + 0) + k * LDA];
                register double a1k = A[(m + 1) + k * LDA];
                register double a2k = A[(m + 2) + k * LDA];
                register double a3k = A[(m + 3) + k * LDA];

                c00 += a0k * B[k + (n + 0) * LDB];
                c01 += a0k * B[k + (n + 1) * LDB];
                c02 += a0k * B[k + (n + 2) * LDB];
                c03 += a0k * B[k + (n + 3) * LDB];

                c10 += a1k * B[k + (n + 0) * LDB];
                c11 += a1k * B[k + (n + 1) * LDB];
                c12 += a1k * B[k + (n + 2) * LDB];
                c13 += a1k * B[k + (n + 3) * LDB];

                c20 += a2k * B[k + (n + 0) * LDB];
                c21 += a2k * B[k + (n + 1) * LDB];
                c22 += a2k * B[k + (n + 2) * LDB];
                c23 += a2k * B[k + (n + 3) * LDB];

                c30 += a3k * B[k + (n + 0) * LDB];
                c31 += a3k * B[k + (n + 1) * LDB];
                c32 += a3k * B[k + (n + 2) * LDB];
                c33 += a3k * B[k + (n + 3) * LDB];
            }

            C[(m + 0) + (n + 0) * LDC] = c00;
            C[(m + 0) + (n + 1) * LDC] = c01;
            C[(m + 0) + (n + 2) * LDC] = c02;
            C[(m + 0) + (n + 3) * LDC] = c03;
            C[(m + 1) + (n + 0) * LDC] = c10;
            C[(m + 1) + (n + 1) * LDC] = c11;
            C[(m + 1) + (n + 2) * LDC] = c12;
            C[(m + 1) + (n + 3) * LDC] = c13;
            C[(m + 2) + (n + 0) * LDC] = c20;
            C[(m + 2) + (n + 1) * LDC] = c21;
            C[(m + 2) + (n + 2) * LDC] = c22;
            C[(m + 2) + (n + 3) * LDC] = c23;
            C[(m + 3) + (n + 0) * LDC] = c30;
            C[(m + 3) + (n + 1) * LDC] = c31;
            C[(m + 3) + (n + 2) * LDC] = c32;
            C[(m + 3) + (n + 3) * LDC] = c33;
        }
    }

    if (M4 == M0 && N4 == N0) return;
    if (M4 != M0) edge_block(M0 - M4, N0, K0, A + M4, LDA, B, LDB, C + M4, LDC);
    if (N4 != N0) edge_block(M0, N0 - N4, K0, A, LDA, B + N4 * LDB, LDB, C + LDC * N4, LDC);
}

void macro_kernel(int M, int N, int K, double *restrict A, int LDA, double *restrict B, int LDB, double *restrict C, int LDC)
{
    micro_kernel(M, N, K, A, LDA, B, LDB, C, LDC);
}

#define M_BLOCK_SIZE 160
#define N_BLOCK_SIZE 160
#define K_BLOCK_SIZE 160

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