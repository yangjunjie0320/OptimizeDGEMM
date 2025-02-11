#include <stdlib.h>

#define M_MACRO_SIZE  384
#define K_MACRO_SIZE  384
#define N_MACRO_SIZE  4096

#define M_MICRO_SIZE  4
#define N_MICRO_SIZE  4

static void pack_A(int M, int K, const double *A, int LDA, double *buffer)
{
    // int incColA = LDA;
    // int incRowA = 1;

    int MP = M / M_MICRO_SIZE;
    int MR = M % M_MICRO_SIZE;

    // Pack complete M_MICRO_SIZE x kc blocks
    for (int mp = 0; mp < MP; ++mp) {
        // Pack one M_MICRO_SIZE x kc block
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < M_MICRO_SIZE; ++m) {
                buffer[m] = A[m + k * LDA];
            }
            buffer += M_MICRO_SIZE;
        }
        A += M_MICRO_SIZE; // Move to next block of rows
    }
    
    // Handle remaining rows with padding
    if (MR > 0) {
        for (int k = 0; k < K; ++k) {
            // Copy actual elements
            for (int m = 0; m < MR; ++m) {
                buffer[m] = A[m];
            }
            // Pad with zeros
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
    // int incColB = LDB;
    // int incRowB = 1;

    int NP = N / N_MICRO_SIZE;
    int NR = N % N_MICRO_SIZE;

    // Pack complete kc x N_MICRO_SIZE blocks
    for (int np = 0; np < NP; ++np) {
        // Pack one kc x N_MICRO_SIZE block
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N_MICRO_SIZE; ++n) {
                buffer[n] = B[k + n * LDB];
            }
            buffer += N_MICRO_SIZE;
        }
        B += LDB * N_MICRO_SIZE; // Move to next block of columns
    }
    
    // Handle remaining columns with padding
    if (NR > 0) {
        for (int k = 0; k < K; ++k) {
            // Copy actual elements
            for (int n = 0; n < NR; ++n) {
                buffer[n] = B[k + n * LDB];
            }
            // Pad with zeros
            for (int n = NR; n < N_MICRO_SIZE; ++n) {
                buffer[n] = 0.0;
            }
            buffer += N_MICRO_SIZE;
            B += 1;
        }
    }
}

//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(int kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC)
{
    double AB[M_MICRO_SIZE*N_MICRO_SIZE];

    int i, j, l;

//
//  Compute AB = A*B
//
    for (l=0; l<M_MICRO_SIZE*N_MICRO_SIZE; ++l) {
        AB[l] = 0;
    }
    for (l=0; l<kc; ++l) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                AB[i+j*M_MICRO_SIZE] += A[i]*B[j];
            }
        }
        A += M_MICRO_SIZE;
        B += N_MICRO_SIZE;
    }

//
//  Update C <- beta*C
//
    if (beta==0.0) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

//
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    if (alpha==1.0) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*M_MICRO_SIZE];
            }
        }
    } else {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*M_MICRO_SIZE];
            }
        }
    }
}

//
//  Compute Y += alpha*X
//
static void
dgeaxpy(int           m,
        int           n,
        double        alpha,
        const double  *X,
        int           incRowX,
        int           incColX,
        double        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int     m,
        int     n,
        double  alpha,
        double  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   double  alpha,
                   double  beta,
                   double  *A,
                   double  *B,
                   double  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+M_MICRO_SIZE-1) / M_MICRO_SIZE;
    int np = (nc+N_MICRO_SIZE-1) / N_MICRO_SIZE;

    int _mr = mc % M_MICRO_SIZE;
    int _nr = nc % N_MICRO_SIZE;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? N_MICRO_SIZE : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? M_MICRO_SIZE : _mr;

            if (mr==M_MICRO_SIZE && nr==N_MICRO_SIZE) {
                dgemm_micro_kernel(kc, alpha, &A[i*kc*M_MICRO_SIZE], &B[j*kc*N_MICRO_SIZE],
                                   beta,
                                   &C[i*M_MICRO_SIZE*incRowC+j*N_MICRO_SIZE*incColC],
                                   incRowC, incColC);
            } else {
                (void *) A;
                (void *) B;
                (void *) C;
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

    int MP_MACRO = (M + M_MACRO_SIZE - 1) / M_MACRO_SIZE;
    int NP_MACRO = (N + N_MACRO_SIZE - 1) / N_MACRO_SIZE;
    int KP_MACRO = (K + K_MACRO_SIZE - 1) / K_MACRO_SIZE;

    int MR_MACRO = M % M_MACRO_SIZE;
    int NR_MACRO = N % N_MACRO_SIZE;
    int KR_MACRO = K % K_MACRO_SIZE;

    for (int np = 0; np < NP_MACRO; ++np) {
        int nr = (np != NP_MACRO - 1 || NR_MACRO == 0) ? N_MACRO_SIZE : NR_MACRO;
        int n = np * N_MACRO_SIZE;

        for (int kp = 0; kp < KP_MACRO; ++kp) {
            int kr = (kp != KP_MACRO - 1 || KR_MACRO == 0) ? K_MACRO_SIZE : KR_MACRO;
            int k = kp * K_MACRO_SIZE;

            const double* B0 = B + k + n * LDB;
            pack_B(kr, nr, B0, LDB, B1);

            for (int mp = 0; mp < MP_MACRO; ++mp) {
                int mr = (mp != MP_MACRO - 1 || MR_MACRO == 0) ? M_MACRO_SIZE : MR_MACRO;
                int m = mp * M_MACRO_SIZE;

                const double* A0 = A + m + k * LDA;
                pack_A(mr, kr, A0, LDA, A1);

                double* C0 = C + m + n * LDC;
                dgemm_macro_kernel(
                    mr, nr, kr,
                    1.0, 0.0,
                    A1, B1,
                    C + m + n * LDC,
                    1, LDC
                );
            }
        }
    }
}