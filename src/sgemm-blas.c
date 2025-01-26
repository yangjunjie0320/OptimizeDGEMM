// #include <cblas.h>
extern void sgemm_(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*); 

/* 
 * This routine performs a sgemm operation
 *  C := C + A * B
 * where A is m-by-k, B is k-by-n, and C is m-by-n.
 * On exit, A and B maintain their input values.    
 * This function wraps a call to the BLAS-3 routine sgemm, 
 * via the standard FORTRAN interface - hence the reference semantics. 
*/
void sgemm_blas(int m, int n, int k, float* a, float* b, float* c)
{
    char layout = 'C';
    char transa = 'N';
    char transb = 'N';
    float alpha = 1.0;
    float beta = 1.0;
    int lda = k;
    int ldb = n;
    int ldc = n;
    sgemm_(
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
