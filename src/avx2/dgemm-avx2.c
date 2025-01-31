#include "immintrin.h"
#define BLOCK_SIZE 4

void edge_block(int m, int n, int l, int lda, int ldb, int ldc, double* a, double* b, double* c) {
    int i, j, k;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double cij = c[i + j * ldc];

            for (k = 0; k < l; k++) {
                double aik = a[i + k * lda];
                double bkj = b[k + j * ldb];
                cij += aik * bkj;
            }
            c[i + j * ldc] = cij;
        }
    }
}

void dgemm(int m, int n, int l, double* a, double* b, double* c)
{
    // A is m x l
    // B is l x n
    // C is m x n
    int lda = m;
    int ldb = l;
    int ldc = m;

    int i, j, k;

    int m1 = m, n1 = n, l1 = l;
    int m2 = m1 & -BLOCK_SIZE;
    int n2 = n1 & -BLOCK_SIZE;

    for (i = 0; i < m2; i += BLOCK_SIZE) {
        for (j = 0; j < n2; j += BLOCK_SIZE) {
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();

            for (k = 0; k < l; k++) {
                __m256d a0 = _mm256_loadu_pd(&a[i + k * lda]);
                __m256d b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                __m256d b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                __m256d b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                __m256d b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);
                
                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
            }

            _mm256_storeu_pd(&c[i + (j + 0) * ldc], _mm256_add_pd(c0, _mm256_loadu_pd(&c[i + (j + 0) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 1) * ldc], _mm256_add_pd(c1, _mm256_loadu_pd(&c[i + (j + 1) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 2) * ldc], _mm256_add_pd(c2, _mm256_loadu_pd(&c[i + (j + 2) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 3) * ldc], _mm256_add_pd(c3, _mm256_loadu_pd(&c[i + (j + 3) * ldc])));
        }
    }

    if (m2 == m1 && n2 == n1) return;

    double *aa, *bb, *cc;

    // case 1: m2 != m1 
    aa = a + m2; bb = b; cc = c + m2;
    if (m2 != m1) edge_block(m1 - m2, n1, l1, lda, ldb, ldc, aa, bb, cc);

    // case 2: n2 != n1
    aa = a; bb = b + n2 * ldb; cc = c + ldc * n2;
    if (n2 != n1) edge_block(m1, n1 - n2, l1, lda, ldb, ldc, aa, bb, cc);
}
