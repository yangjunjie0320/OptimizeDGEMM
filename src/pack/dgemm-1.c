#include <stdlib.h>

#define M_MACRO_SIZE  384
#define K_MACRO_SIZE  384
#define N_MACRO_SIZE  4096

#define M_MICRO_SIZE  4
#define N_MICRO_SIZE  4

#define min(a, b) ((a) < (b) ? (a) : (b))

//
//  Packing panels from A with padding if required
//
static void pack_A(int M, int K, const double *A, int LDA, double *buffer)
{
    int MP = M / M_MICRO_SIZE;
    int MR = M % M_MICRO_SIZE;

    for (int mp = 0; mp < MP; ++mp) {
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < M_MICRO_SIZE; ++m) {
                buffer[m] = A[m + k * LDA];
            }
            buffer += M_MICRO_SIZE;
        }
        A += M_MICRO_SIZE * LDA;
    }

    if (MR > 0) {
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < MR; ++m) {
                buffer[m] = A[m + k * LDA];
            }
            for (int m = MR; m < M_MICRO_SIZE; ++m) {
                buffer[m] = 0.0;
            }
            buffer += M_MICRO_SIZE;
            // A += LDA;
        }
    }
}

static void pack_B(int K, int N, const double *B, int LDB, double *buffer)
{
    int NP = N / N_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    for (int np = 0; np < NP; ++np) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N_MICRO_SIZE; ++n) {
                buffer[n] = B[k + n * LDB];
            }
            buffer += N_MICRO_SIZE;
        }
        B += N_MICRO_SIZE * LDB;
    }

    if (NR > 0) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < NR; ++n) {
                buffer[n] = B[k + n * LDB];
            }
            for (int n = NR; n < N_MICRO_SIZE; ++n) {
                buffer[n] = 0.0;
            }
            buffer += N_MICRO_SIZE;
        }
    }
}

static void micro_kernel(int M, int N, int K, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{
    double AB[M_MICRO_SIZE * N_MICRO_SIZE];

    for (int mn = 0; mn < M_MICRO_SIZE * N_MICRO_SIZE; ++mn) {
        AB[mn] = 0.0;
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N_MICRO_SIZE; ++n) {
            for (int m = 0; m < M_MICRO_SIZE; ++m) {
                AB[m + n * M_MICRO_SIZE] += A[m] * B[n];
            }
        }
        A += M_MICRO_SIZE;
        B += N_MICRO_SIZE;
    }

    for (int n = 0; n < N_MICRO_SIZE; ++n) {
        for (int m = 0; m < M_MICRO_SIZE; ++m) {
            C[m + n * LDC] += AB[m + n * M_MICRO_SIZE];
        }
    }
}

static void macro_kernel(int M, int N, int K, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{
    int MP = (M + M_MICRO_SIZE - 1) / M_MICRO_SIZE;
    int NP = (N + N_MICRO_SIZE - 1) / N_MICRO_SIZE;

    int MR = M % M_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    for (int np = 0; np < NP; ++np) {
        int nr = (np != NP - 1 || NR == 0) ? N_MICRO_SIZE : NR;
        int n = np * N_MICRO_SIZE;

        for (int mp = 0; mp < MP; ++mp) {
            int mr = (mp != MP - 1 || MR == 0) ? M_MICRO_SIZE : MR;
            int m = mp * M_MICRO_SIZE;

            if (mr == M_MICRO_SIZE && nr == N_MICRO_SIZE) {
                micro_kernel(mr, nr, K, A, LDA, B, LDB, C, LDC);
            } else {
                // deal with edge blocks
                const double *A1 = A + m * K;
                const double *B1 = B + n * K;
                double *C1 = C + m + n * LDC;

                for (int mm = 0; mm < mr; mm++) {
                    for (int nn = 0; nn < nr; nn++) {
                        double cmn = 0.0;
                        for (int kk = 0; kk < K; kk++) {
                            double amk = A1[mm + kk * MR];
                            double bkn = B1[kk + nn * NR];
                            cmn += amk * bkn;
                        }
                        C1[mm + nn * LDC] = cmn;
                    }
                }
            }
        }
    }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C)
{
    int LDA = M;
    int LDB = K;
    int LDC = M;

    double *A1 = (double *) malloc(M_MACRO_SIZE * K_MACRO_SIZE * sizeof(double));
    double *B1 = (double *) malloc(K_MACRO_SIZE * N_MACRO_SIZE * sizeof(double));

    int MP = (M + M_MACRO_SIZE - 1) / M_MACRO_SIZE;
    int NP = (N + N_MACRO_SIZE - 1) / N_MACRO_SIZE;
    int KP = (K + K_MACRO_SIZE - 1) / K_MACRO_SIZE;

    int MR = M % M_MACRO_SIZE;
    int NR = N % N_MACRO_SIZE;
    int KR = K % K_MACRO_SIZE;

    for (int np = 0; np < NP; ++np) {
        int nr = (np != NP - 1 || NR == 0) ? N_MACRO_SIZE : NR;
        int n = np * N_MACRO_SIZE;

        for (int kp = 0; kp < KP; ++kp) {
            int kr = (kp != KP - 1 || KR == 0) ? K_MACRO_SIZE : KR;
            int k = kp * K_MACRO_SIZE;

            const double* B0 = B + k + n * LDB;
            pack_B(kr, nr, B0, LDB, B1);

            for (int mp = 0; mp < MP; ++mp) {
                int mr = (mp != MP - 1 || MR == 0) ? M_MACRO_SIZE : MR;
                int m = mp * M_MACRO_SIZE;

                const double* A0 = A + m + k * LDA;
                pack_A(mr, kr, A0, LDA, A1);

                double* C0 = C + m + n * LDC;
                macro_kernel(
                    mr, nr, kr,
                    A1, LDA,
                    B1, LDB,
                    C0, LDC
                );
            }
        }
    }

    free(A1);
    free(B1);
}