extern void dgemm_(const char*, const char*, const int*, const int*, const int*, 
                  const double*, const double*, const int*, 
                  const double*, const int*, 
                  double*, double*, const int*);

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
        (int*) &m, (int*) &n, (int*) &k,
        &alpha,
        (double*) a, &lda,
        (double*) b, &ldb,
        &beta, c, &ldc
    );
}