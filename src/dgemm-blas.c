extern void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void dgemm_blas(const int m, const int n, const int k, const double* a, const double* b, double* c)
{
    // in column major order
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int lda = m;
    int ldb = k;
    int ldc = m;
    dgemm_(
        &transa,  // transa
        &transb,  // transb
        &m, &n, &k,
        &alpha,
        a, &lda,
        b, &ldb,
        &beta,
        c, &ldc
    );
}