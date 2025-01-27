extern void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 

void dgemm_blas(int m, int n, int k, double* a, double* b, double* c)
{
    char layout = 'C';
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 1.0;
    int lda = k;
    int ldb = n;
    int ldc = n;
    dgemm_(
      // layout, 
      &transa, 
      &transb, 
      &m, &n, &k, 
      &alpha, 
      a, &lda, 
      b, &ldb, 
      &beta, 
      c, &ldc
    );
}   
