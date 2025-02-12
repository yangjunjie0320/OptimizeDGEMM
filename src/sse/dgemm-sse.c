#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>

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

static void
dgemm_micro_kernel(int kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC)
{
    const int MR = M_MICRO_SIZE;
    const int NR = N_MICRO_SIZE;
    double AB[MR*NR] __attribute__ ((aligned (16)));

    // Cols of AB in SSE registers
    __m128d   ab_00_10, ab_20_30;
    __m128d   ab_01_11, ab_21_31;
    __m128d   ab_02_12, ab_22_32;
    __m128d   ab_03_13, ab_23_33;

    __m128d   a_01, a_23;
    __m128d   b_00, b_11, b_22, b_33;
    __m128d   tmp1, tmp2;

    int i, j, l;

    ab_00_10 = _mm_setzero_pd();
    ab_20_30 = _mm_setzero_pd();
    ab_01_11 = _mm_setzero_pd();
    ab_21_31 = _mm_setzero_pd();
    ab_02_12 = _mm_setzero_pd();
    ab_22_32 = _mm_setzero_pd();
    ab_03_13 = _mm_setzero_pd();
    ab_23_33 = _mm_setzero_pd();

//
//  Compute AB = A*B
//
    for (l=0; l<kc; ++l) {
        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A+2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B+1);
        b_22 = _mm_load_pd1(B+2);
        b_33 = _mm_load_pd1(B+3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;
    }

    _mm_store_pd(AB+ 0, ab_00_10);
    _mm_store_pd(AB+ 2, ab_20_30);

    _mm_store_pd(AB+ 4, ab_01_11);
    _mm_store_pd(AB+ 6, ab_21_31);

    _mm_store_pd(AB+ 8, ab_02_12);
    _mm_store_pd(AB+10, ab_22_32);

    _mm_store_pd(AB+12, ab_03_13);
    _mm_store_pd(AB+14, ab_23_33);

//
//  Update C <- beta*C
//
    if (beta==0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

//
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    if (alpha==1.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
            }
        }
    }
}

//  Micro kernel for multiplying panels from A and B.
static void micro_kernel(int M, int N, int K, double* A, int LDA, double* B, int LDB, double* C, int LDC)
{
    dgemm_micro_kernel(K, 1.0, A, B, 1.0, C, 1, LDC);
}

//  Macro Kernel for the multiplication of blocks of A and B.
static void macro_kernel(int M, int N, int K, double* A, int LDA, double* B, int LDB, double* C, int LDC)
{
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

    // double *A1 = (double *) malloc(M_MACRO_SIZE * K_MACRO_SIZE * sizeof(double));
    // double *B1 = (double *) malloc(K_MACRO_SIZE * N_MACRO_SIZE * sizeof(double));
    double *A1 = (double *) aligned_alloc(16, M_MACRO_SIZE * K_MACRO_SIZE * sizeof(double));
    double *B1 = (double *) aligned_alloc(16, K_MACRO_SIZE * N_MACRO_SIZE * sizeof(double));

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