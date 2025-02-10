#define M_BLOCK_SIZE  384
#define K_BLOCK_SIZE  384
#define N_BLOCK_SIZE  4096

#define BLOCK_SIZE  4
#define BLOCK_SIZE  4

#define min(a, b) ((a) < (b) ? (a) : (b))



static void pack_A(int mc, int kc, const double *A, int LDA, double *buffer)
{
    int mp  = mc / BLOCK_SIZE;
    int _mr = mc % BLOCK_SIZE;

    int i, j;

    for (i=0; i<mp; ++i) {
        for (j=0; j<kc; ++j) {
            for (int ii=0; ii<BLOCK_SIZE; ++ii) {
                buffer[ii] = A[i*BLOCK_SIZE*LDA + ii + j*LDA];
            }
            buffer += BLOCK_SIZE;
        }
    }
    if (_mr > 0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[(mp*BLOCK_SIZE+i) + j*LDA];
            }
            for (i=_mr; i<BLOCK_SIZE; ++i) {
                buffer[i] = 0.0;
            }
            buffer += BLOCK_SIZE;
        }
    }
}

static void pack_B(int kc, int nc, const double *B, int LDB, double *buffer)
{
    int np  = nc / BLOCK_SIZE;
    int _nr = nc % BLOCK_SIZE;

    int i, j;

    for (j=0; j<np; ++j) {
        for (i=0; i<kc; ++i) {
            for (int jj=0; jj<BLOCK_SIZE; ++jj) {
                buffer[jj] = B[i + (j*BLOCK_SIZE+jj)*LDB];
            }
            buffer += BLOCK_SIZE;
        }
    }
    if (_nr > 0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[i + (np*BLOCK_SIZE+j)*LDB];
            }
            for (j=_nr; j<BLOCK_SIZE; ++j) {
                buffer[j] = 0.0;
            }
            buffer += BLOCK_SIZE;
        }
    }
}

static void dgemm_micro_kernel(int kc, double *A, double *B, double *C, int LDC)
{
    int i, j, l;

    for (l=0; l<kc; ++l) {
        for (j=0; j<BLOCK_SIZE; ++j) {
            for (i=0; i<BLOCK_SIZE; ++i) {
                C[i + j*LDC] += A[i] * B[j];
            }
        }
        A += BLOCK_SIZE;
        B += BLOCK_SIZE;
    }
}

static void dgemm_macro_kernel(int M, int N, int K, double *A, double *B, double *C, int LDC)
{
    int mp = (M+BLOCK_SIZE-1) / BLOCK_SIZE;
    int np = (N+BLOCK_SIZE-1) / BLOCK_SIZE;

    int _mr = M % BLOCK_SIZE;
    int _nr = N % BLOCK_SIZE;

    int mr, nr;
    int i, j;

    int m_inc, n_inc, k_inc;
    for (int m = 0; m < M;) {
        m_inc = min(M_BLOCK_SIZE, M - m);

        for (int n = 0; n < N;) {
            n_inc = min(N_BLOCK_SIZE, N - n);

            if (m_inc == M_BLOCK_SIZE && n_inc == N_BLOCK_SIZE) {
                dgemm_micro_kernel(
                    K, &A[m * M_BLOCK_SIZE + k * K_BLOCK_SIZE * M], 
                    &B[n * N_BLOCK_SIZE + k * K_BLOCK_SIZE * N], 
                    &C[m * M_BLOCK_SIZE + n * N_BLOCK_SIZE * LDC], 
                    LDC
                    );
            } else {
                for (int k=0; k<K; ++k) {
                    for (int n=0; n<n_inc; ++n) {
                        for (int m=0; m<m_inc; ++m) {
                            C[m * M_BLOCK_SIZE + n * N_BLOCK_SIZE * LDC] += A[m * M_BLOCK_SIZE + k * K_BLOCK_SIZE * M] * B[n * N_BLOCK_SIZE + k * K_BLOCK_SIZE * N];
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

    double *A_packed = (double *) malloc(M_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(double));
    double *B_packed = (double *) malloc(K_BLOCK_SIZE * N_BLOCK_SIZE * sizeof(double));

    int m_inc, n_inc, k_inc;
    for (int k = 0; k < K;) {
        k_inc = min(K_BLOCK_SIZE, K - k);

        for (int n = 0; n < N;) {
            n_inc = min(N_BLOCK_SIZE, N - n);
            pack_B(kc, nc, &B[n * N_BLOCK_SIZE + k * K_BLOCK_SIZE * K], K, B_packed);

            for (int m = 0; m < M;) {
                m_inc = min(M_BLOCK_SIZE, M - m);
                pack_A(m_inc, kc, &A[m * M_BLOCK_SIZE + k * K_BLOCK_SIZE * M], M, B_packed);

                macro_kernel(
                    m_inc, n_inc, k_inc,
                    A_packed, B_packed,
                    C + m + n * LDC, LDC
                );
                m += m_inc;
            }
            n += n_inc;
        }
        k += k_inc;
    }
}
