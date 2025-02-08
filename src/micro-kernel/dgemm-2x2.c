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

#define BLOCK_SIZE 2
void micro_kernel(int M, int N, int K, double *restrict A, int LDA, double *restrict B, int LDB, double *restrict C, int LDC)
{
    int M0 = M, N0 = N, K0 = K;
    int M2 = M0 & -BLOCK_SIZE;
    int N2 = N0 & -BLOCK_SIZE;
    int K2 = K0 & -BLOCK_SIZE;

    register double c00, c01, c10, c11;

    for (int m = 0; m < M2; m += BLOCK_SIZE) {
        for (int n = 0; n < N2; n += BLOCK_SIZE) {
            c00 = C[(m + 0) + (n + 0) * LDC];
            c01 = C[(m + 0) + (n + 1) * LDC];
            c10 = C[(m + 1) + (n + 0) * LDC];
            c11 = C[(m + 1) + (n + 1) * LDC];

            for (int k = 0; k < K2; k++) {
                double a0k = A[(m + 0) + k * LDA];
                double a1k = A[(m + 1) + k * LDA];

                c00 += a0k * B[k + (n + 0) * LDB];
                c01 += a0k * B[k + (n + 1) * LDB];
                c10 += a1k * B[k + (n + 0) * LDB];
                c11 += a1k * B[k + (n + 1) * LDB];
            }

            C[(m + 0) + (n + 0) * LDC] = c00;
            C[(m + 0) + (n + 1) * LDC] = c01;
            C[(m + 1) + (n + 0) * LDC] = c10;
            C[(m + 1) + (n + 1) * LDC] = c11;
        }
    }

    if (M2 == M0 && N2 == N0) return;
    if (M2 != M0) edge_block(M0 - M2, N0, K0, A + M2, LDA, B, LDB, C + M2, LDC);
    if (N2 != N0) edge_block(M0, N0 - N2, K0, A, LDA, B + N2 * LDB, LDB, C + LDC * N2, LDC);
}

void dgemm(int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    int LDA = M;
    int LDB = K;
    int LDC = M;

    micro_kernel(M, N, K, A, LDA, B, LDB, C, LDC);
}
