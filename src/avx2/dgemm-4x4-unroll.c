#include "immintrin.h"
#define BLOCK_SIZE 4
#define UNROLL 4

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

    int m0 = m, n0 = n, l0 = l;
    int m4 = m0 & -BLOCK_SIZE;
    int n4 = n0 & -BLOCK_SIZE;
    int l4 = l0 & -BLOCK_SIZE;

    for (i = 0; i < m4; i += BLOCK_SIZE) {
        for (j = 0; j < n4; j += BLOCK_SIZE) {
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();

            __m256d a0, b0, b1, b2, b3;

            for (k = 0; k < l4;) {
                a0 = _mm256_loadu_pd(&a[i + k * lda]);
                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                k++;

                a0 = _mm256_loadu_pd(&a[i + k * lda]);
                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                k++;

                a0 = _mm256_loadu_pd(&a[i + k * lda]);
                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                k++;

                a0 = _mm256_loadu_pd(&a[i + k * lda]);
                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                k++;
            }

            for (k = l4; k < l; ) {
                a0 = _mm256_loadu_pd(&a[i + k * lda]);
                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c0 = _mm256_fmadd_pd(a0, b0, c0);
                c1 = _mm256_fmadd_pd(a0, b1, c1);
                c2 = _mm256_fmadd_pd(a0, b2, c2);
                c3 = _mm256_fmadd_pd(a0, b3, c3);
                k++;
            }

            _mm256_storeu_pd(&c[i + (j + 0) * ldc], _mm256_add_pd(c0, _mm256_loadu_pd(&c[i + (j + 0) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 1) * ldc], _mm256_add_pd(c1, _mm256_loadu_pd(&c[i + (j + 1) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 2) * ldc], _mm256_add_pd(c2, _mm256_loadu_pd(&c[i + (j + 2) * ldc])));
            _mm256_storeu_pd(&c[i + (j + 3) * ldc], _mm256_add_pd(c3, _mm256_loadu_pd(&c[i + (j + 3) * ldc])));
        }
    }

    if (m4 == m0 && n4 == n0) return;

    double *aa, *bb, *cc;

    // case 1: m4 != m0 
    aa = a + m4; bb = b; cc = c + m4;
    if (m4 != m0) edge_block(m0 - m4, n0, l0, lda, ldb, ldc, aa, bb, cc);

    // case 2: n4 != n0
    aa = a; bb = b + n4 * ldb; cc = c + ldc * n4;
    if (n4 != n0) edge_block(m0, n0 - n4, l0, lda, ldb, ldc, aa, bb, cc);
}
