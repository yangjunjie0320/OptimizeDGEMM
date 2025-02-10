#include <string.h>

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

    int i, j;
    for (int p = 0; p < MP; ++p) {
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
            A += LDA;
        }
    }
}

static void pack_B(int K, int N, const double *B, int LDB, double *buffer)
{
    int NP = N / N_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    int i, j;
    
    for (int p = 0; p < NP; ++p) {
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

static void dgemm_micro_kernel(int K, double *A, double *B, double *C, int LDC)
{
    double AB[M_MICRO_SIZE * N_MICRO_SIZE];

    for (int x = 0; x < M_MICRO_SIZE * N_MICRO_SIZE; ++x) {
        AB[x] = 0.0;
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

static void dgemm_macro_kernel(int M, int N, int K, double *A, double *B, double *C, int LDC)
{
    // int mp = (M+M_MICRO_SIZE-1) / M_MICRO_SIZE;
    // int np = (N+M_MICRO_SIZE-1) / M_MICRO_SIZE;

    // int _mr = M % M_MICRO_SIZE;
    // int _nr = N % M_MICRO_SIZE;

    int MP = M / M_MICRO_SIZE;
    int MR = M % M_MICRO_SIZE;

    int NP = N / N_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    int mr, nr;
    int i, j;

    int m_inc, n_inc, k_inc;
    for (int m = 0; m < M;) {
        m_inc = min(M_MACRO_SIZE, M - m);

        for (int n = 0; n < N;) {
            n_inc = min(N_MACRO_SIZE, N - n);

            if (m_inc == M_MACRO_SIZE && n_inc == N_MACRO_SIZE) {
                dgemm_micro_kernel(
                    K, &A[m * M_MACRO_SIZE + k * K_MACRO_SIZE * M], 
                    &B[n * N_MACRO_SIZE + k * K_MACRO_SIZE * N], 
                    &C[m * M_MACRO_SIZE + n * N_MACRO_SIZE * LDC], 
                    LDC
                    );
            } else {
                for (int k=0; k<K; ++k) {
                    for (int n=0; n<n_inc; ++n) {
                        for (int m=0; m<m_inc; ++m) {
                            C[m * M_MACRO_SIZE + n * N_MACRO_SIZE * LDC] += A[m * M_MACRO_SIZE + k * K_MACRO_SIZE * M] * B[n * N_MACRO_SIZE + k * K_MACRO_SIZE * N];
                        }
                    }
                }
            }
        }
    }
}

void dgemm(int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    int LDA = M;
    int LDB = K;
    int LDC = M;

    double *A1 = (double *) malloc(M_MACRO_SIZE * K_MACRO_SIZE * sizeof(double));
    double *B1 = (double *) malloc(K_MACRO_SIZE * N_MACRO_SIZE * sizeof(double));

    int m_inc, n_inc, k_inc;
    for (int k = 0; k < K;) {
        k_inc = min(K_MACRO_SIZE, K - k);

        for (int n = 0; n < N;) {
            n_inc = min(N_MACRO_SIZE, N - n);
            double* B0 = B + k * K_MACRO_SIZE * N;
            pack_B(k_inc, n_inc, B0, K, B1);

            for (int m = 0; m < M;) {
                m_inc = min(M_MACRO_SIZE, M - m);
                double* A0 = A + m * M_MACRO_SIZE + k * K_MACRO_SIZE * M;
                pack_A(m_inc, k_inc, A0, M, A1);

                dgemm_macro_kernel(
                    m_inc, n_inc, k_inc,
                    A1, B1,
                    C + m + n * LDC, LDC
                );
                m += m_inc;
            }
            n += n_inc;
        }
        k += k_inc;
    }
}
