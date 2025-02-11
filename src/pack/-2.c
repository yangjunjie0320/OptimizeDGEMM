#define M_MACRO_SIZE  384
#define K_MACRO_SIZE  384
#define N_MACRO_SIZE  4096

#define M_MICRO_SIZE  4
#define N_MICRO_SIZE  4

//
//  Local buffers for storing panels from A, B and C
//
static double _A[M_MACRO_SIZE*K_MACRO_SIZE];
static double _B[K_MACRO_SIZE*N_MACRO_SIZE];
static double _C[M_MICRO_SIZE*N_MICRO_SIZE];

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const double *A, int 1, int LDA,
          double *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<M_MICRO_SIZE; ++i) {
            buffer[i] = A[i*1];
        }
        buffer += M_MICRO_SIZE;
        A      += LDA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const double *A, int 1, int LDA,
       double *buffer)
{
    int mp  = mc / M_MICRO_SIZE;
    int _mr = mc % M_MICRO_SIZE;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, 1, LDA, buffer);
        buffer += kc*M_MICRO_SIZE;
        A      += M_MICRO_SIZE*1;
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*1];
            }
            for (i=_mr; i<M_MICRO_SIZE; ++i) {
                buffer[i] = 0.0;
            }
            buffer += M_MICRO_SIZE;
            A      += LDA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const double *B, int 1, int LDB,
          double *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            buffer[j] = B[j*LDB];
        }
        buffer += N_MICRO_SIZE;
        B      += 1;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const double *B, int 1, int LDB,
       double *buffer)
{
    int np  = nc / N_MICRO_SIZE;
    int _nr = nc % N_MICRO_SIZE;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, 1, LDB, buffer);
        buffer += kc*N_MICRO_SIZE;
        B      += N_MICRO_SIZE*LDB;
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*LDB];
            }
            for (j=_nr; j<N_MICRO_SIZE; ++j) {
                buffer[j] = 0.0;
            }
            buffer += N_MICRO_SIZE;
            B      += 1;
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
                   double *C, int 1, int LDC)
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
                C[i*1+j*LDC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*1+j*LDC] *= beta;
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
                C[i*1+j*LDC] += AB[i+j*M_MICRO_SIZE];
            }
        }
    } else {
        for (j=0; j<N_MICRO_SIZE; ++j) {
            for (i=0; i<M_MICRO_SIZE; ++i) {
                C[i*1+j*LDC] += alpha*AB[i+j*M_MICRO_SIZE];
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
                   double  *C,
                   int     1,
                   int     LDC)
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
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*M_MICRO_SIZE], &_B[j*kc*N_MICRO_SIZE],
                                   beta,
                                   &C[i*M_MICRO_SIZE*1+j*N_MICRO_SIZE*LDC],
                                   1, LDC);
            } else {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*M_MICRO_SIZE], &_B[j*kc*N_MICRO_SIZE],
                                   0.0,
                                   _C, 1, M_MICRO_SIZE);
                dgescal(mr, nr, beta,
                        &C[i*M_MICRO_SIZE*1+j*N_MICRO_SIZE*LDC], 1, LDC);
                dgeaxpy(mr, nr, 1.0, _C, 1, M_MICRO_SIZE,
                        &C[i*M_MICRO_SIZE*1+j*N_MICRO_SIZE*LDC], 1, LDC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void
dgemm_nn(int            m,
         int            n,
         int            k,
         double         alpha,
         const double   *A,
         int            1,
         int            LDA,
         const double   *B,
         int            1,
         int            LDB,
         double         beta,
         double         *C,
         int            1,
         int            LDC)
{
    int mb = (m+M_MACRO_SIZE-1) / M_MACRO_SIZE;
    int nb = (n+N_MACRO_SIZE-1) / N_MACRO_SIZE;
    int kb = (k+K_MACRO_SIZE-1) / K_MACRO_SIZE;

    int _mc = m % M_MACRO_SIZE;
    int _nc = n % N_MACRO_SIZE;
    int _kc = k % K_MACRO_SIZE;

    int mc, nc, kc;
    int i, j, l;

    double _beta;

    if (alpha==0.0 || k==0) {
        dgescal(m, n, beta, C, 1, LDC);
        return;
    }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? N_MACRO_SIZE : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? K_MACRO_SIZE   : _kc;
            _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l*K_MACRO_SIZE*1+j*N_MACRO_SIZE*LDB], 1, LDB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? M_MACRO_SIZE : _mc;

                pack_A(mc, kc,
                       &A[i*M_MACRO_SIZE*1+l*K_MACRO_SIZE*LDA], 1, LDA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc, alpha, _beta,
                                   &C[i*M_MACRO_SIZE*1+j*N_MACRO_SIZE*LDC],
                                   1, LDC);
            }
        }
    }
}

void dgemm(int M, int N, int K, double *A, double *B, double *C) {
    int LDA = M;
    int LDB = K;
    int LDC = M;

    // in column major order
    // int 1 = 1;
    // int LDA = LDA;
    // int 1 = 1;
    // int LDB = LDB;
    // int 1 = 1;
    // int LDC = LDC;

    dgemm_nn(M, N, K, 1.0, A, 1, LDA, B, 1, LDB, 0.0, C, 1, LDC);
}