#include <stdlib.h>
#include <assert.h>
#define M_MACRO_SIZE  384
#define K_MACRO_SIZE  384
#define N_MACRO_SIZE  2048

#define M_MICRO_SIZE  4
#define N_MICRO_SIZE  4

static void pack_A(int M, int K, const double *A, int LDA, double *buffer)
{
    int MP = M / M_MICRO_SIZE;  // Number of complete MR x kc blocks
    int MR = M % M_MICRO_SIZE; // Remaining rows
    int m, mp, k;

    // Pack complete MR x kc blocks
    for (mp = 0; mp < MP; ++mp) {

        // Pack one MR x kc block
        for (k = 0; k < K; ++k) {

            #pragma unroll M_MICRO_SIZE
            for (int m = 0; m < M_MICRO_SIZE; ++m) {
                buffer[m] = A[m];
            }
            buffer += M_MICRO_SIZE;
            A += LDA;
        }
        A = A - K * LDA + M_MICRO_SIZE; // Move to next block of rows
    }
    
    // Handle remaining rows with padding
    if (MR > 0) {
        for (k = 0; k < K; ++k) {
            // Copy actual elements
            for (m = 0; m < MR; ++m) {
                buffer[m] = A[m];
            }
            // Pad with zeros
            for (m = MR; m < M_MICRO_SIZE; ++m) {
                buffer[m] = 0.0;
            }
            buffer += M_MICRO_SIZE;
            A += LDA;
        }
    }
}

static void pack_B(int K, int N, const double *B, int LDB, double *buffer)
{
    int NP = N / N_MICRO_SIZE;  // Number of complete kc x NR blocks
    int NR = N % N_MICRO_SIZE; // Remaining columns
    int n, np, k;

    // Pack complete kc x NR blocks
    for (np = 0; np < NP; ++np) {
        
        // Pack one kc x NR block
        for (k = 0; k < K; ++k) {

            #pragma unroll N_MICRO_SIZE
            for (int n = 0; n < N_MICRO_SIZE; ++n) {
                buffer[n] = B[n * LDB];
            }
            buffer += N_MICRO_SIZE;
            B += 1;
        }
        B = B - K + N_MICRO_SIZE * LDB; // Move to next block of columns
    }
    
    // Handle remaining columns with padding
    if (NR > 0) {
        for (k = 0; k < K; ++k) {
            // Copy actual elements
            for (n = 0; n < NR; ++n) {
                buffer[n] = B[n * LDB];
            }
            // Pad with zeros
            for (n = NR; n < N_MICRO_SIZE; ++n) {
                buffer[n] = 0.0;
            }
            buffer += N_MICRO_SIZE;
            B += 1;
        }
    }
}

//  Micro kernel for multiplying panels from A and B.
static void micro_kernel(int M, int N, int K, double* A, int LDA, double* B, int LDB, double* C, int LDC)
{
    int incRowC = 1;
    int incColC = LDC;
    double alpha = 1.0;
    double beta = 1.0;

    double AB[M_MICRO_SIZE * N_MICRO_SIZE];

    #pragma unroll M_MICRO_SIZE * N_MICRO_SIZE
    for (int mn = 0; mn < M_MICRO_SIZE * N_MICRO_SIZE; ++mn) {
        AB[mn] = 0;
    }

    for (int k = 0; k < K; ++k) {

        #pragma unroll N_MICRO_SIZE
        for (int n = 0; n < N_MICRO_SIZE; ++n) {
            double bkn = B[n + k * N_MICRO_SIZE];

            #pragma unroll M_MICRO_SIZE 
            for (int m = 0; m < M_MICRO_SIZE; ++m) {
                double amk = A[m + k * M_MICRO_SIZE];

                int mn = m + n * M_MICRO_SIZE;
                AB[mn] += amk * bkn;
            }
        }
    }

    #pragma unroll N_MICRO_SIZE
    for (int n = 0; n < N_MICRO_SIZE; ++n) {

        #pragma unroll M_MICRO_SIZE
        for (int m = 0; m < M_MICRO_SIZE; ++m) {
            C[m + n * LDC] += AB[m + n * M_MICRO_SIZE];
        }
    }
}

//  Macro Kernel for the multiplication of blocks of A and B.
static void macro_kernel(int M, int N, int K, double* A, int LDA, double* B, int LDB, double* C, int LDC)
{
    double alpha = 1.0;
    double beta = 1.0;
    int incRowC = 1;

    int MP = (M+M_MICRO_SIZE-1) / M_MICRO_SIZE;
    int NP = (N+N_MICRO_SIZE-1) / N_MICRO_SIZE;

    int MR = M % M_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    for (int np = 0; np < NP; ++np) {
        int nr = (np != NP - 1 || NR == 0) ? N_MICRO_SIZE : NR;
        int n = np * N_MICRO_SIZE;

        for (int mp = 0; mp < MP; ++mp) {
            int mr = (mp != MP - 1 || MR == 0) ? M_MICRO_SIZE : MR;
            int m = mp * M_MICRO_SIZE;

            double* A0 = A + m * K;
            double* B0 = B + n * K;
            double* C0 = C + m + n * LDC;

            assert(mr == M_MICRO_SIZE && nr == N_MICRO_SIZE);
            micro_kernel(mr, nr, K, A0, LDA, B0, LDB, C0, LDC);
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
}