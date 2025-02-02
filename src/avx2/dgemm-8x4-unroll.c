#include "immintrin.h"
#define BLOCK_SIZE_M 8
#define BLOCK_SIZE_N 4
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
    int m8 = m0 & -BLOCK_SIZE_M;
    int n4 = n0 & -BLOCK_SIZE_N;
    int l4 = l0 & -UNROLL;

    for (i = 0; i < m8; i += BLOCK_SIZE_M) {
        for (j = 0; j < n4; j += BLOCK_SIZE_N) {
            __m256d c00 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd();
            __m256d c01 = _mm256_setzero_pd();
            __m256d c11 = _mm256_setzero_pd();
            __m256d c21 = _mm256_setzero_pd();
            __m256d c31 = _mm256_setzero_pd();

            __m256d a0, a1, b0, b1, b2, b3;

            for (k = 0; k < l4;) {
                a0 = _mm256_loadu_pd(&a[(i + 0) + k * lda]);
                a1 = _mm256_loadu_pd(&a[(i + 4) + k * lda]);

                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a1, b0, c01);
                c10 = _mm256_fmadd_pd(a0, b1, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a0, b2, c20);
                c21 = _mm256_fmadd_pd(a1, b2, c21);
                c30 = _mm256_fmadd_pd(a0, b3, c30);
                c31 = _mm256_fmadd_pd(a1, b3, c31);
                k++;

                a0 = _mm256_loadu_pd(&a[(i + 0) + k * lda]);
                a1 = _mm256_loadu_pd(&a[(i + 4) + k * lda]);

                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a1, b0, c01);
                c10 = _mm256_fmadd_pd(a0, b1, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a0, b2, c20);
                c21 = _mm256_fmadd_pd(a1, b2, c21);
                c30 = _mm256_fmadd_pd(a0, b3, c30);
                c31 = _mm256_fmadd_pd(a1, b3, c31);
                k++;

                a0 = _mm256_loadu_pd(&a[(i + 0) + k * lda]);
                a1 = _mm256_loadu_pd(&a[(i + 4) + k * lda]);

                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);   
                
                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a1, b0, c01);
                c10 = _mm256_fmadd_pd(a0, b1, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a0, b2, c20);
                c21 = _mm256_fmadd_pd(a1, b2, c21);
                c30 = _mm256_fmadd_pd(a0, b3, c30);
                c31 = _mm256_fmadd_pd(a1, b3, c31);
                k++;

                a0 = _mm256_loadu_pd(&a[(i + 0) + k * lda]);
                a1 = _mm256_loadu_pd(&a[(i + 4) + k * lda]);

                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a1, b0, c01);
                c10 = _mm256_fmadd_pd(a0, b1, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a0, b2, c20);
                c21 = _mm256_fmadd_pd(a1, b2, c21);
                c30 = _mm256_fmadd_pd(a0, b3, c30);
                c31 = _mm256_fmadd_pd(a1, b3, c31);
                k++;
            }

            for (k = l4; k < l; ) {
                a0 = _mm256_loadu_pd(&a[(i + 0) + k * lda]);
                a1 = _mm256_loadu_pd(&a[(i + 4) + k * lda]);

                b0 = _mm256_broadcast_sd(&b[k + (j + 0) * ldb]);
                b1 = _mm256_broadcast_sd(&b[k + (j + 1) * ldb]);
                b2 = _mm256_broadcast_sd(&b[k + (j + 2) * ldb]);
                b3 = _mm256_broadcast_sd(&b[k + (j + 3) * ldb]);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a1, b0, c01);
                c10 = _mm256_fmadd_pd(a0, b1, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a0, b2, c20);
                c21 = _mm256_fmadd_pd(a1, b2, c21);
                c30 = _mm256_fmadd_pd(a0, b3, c30);
                c31 = _mm256_fmadd_pd(a1, b3, c31);
                k++;
            }
                
            _mm256_storeu_pd(&c[(i + 0) + (j + 0) * ldc], _mm256_add_pd(c00, _mm256_loadu_pd(&c[(i + 0) + (j + 0) * ldc])));
            _mm256_storeu_pd(&c[(i + 4) + (j + 0) * ldc], _mm256_add_pd(c01, _mm256_loadu_pd(&c[(i + 4) + (j + 0) * ldc])));

            _mm256_storeu_pd(&c[(i + 0) + (j + 1) * ldc], _mm256_add_pd(c10, _mm256_loadu_pd(&c[(i + 0) + (j + 1) * ldc])));
            _mm256_storeu_pd(&c[(i + 4) + (j + 1) * ldc], _mm256_add_pd(c11, _mm256_loadu_pd(&c[(i + 4) + (j + 1) * ldc])));

            _mm256_storeu_pd(&c[(i + 0) + (j + 2) * ldc], _mm256_add_pd(c20, _mm256_loadu_pd(&c[(i + 0) + (j + 2) * ldc])));
            _mm256_storeu_pd(&c[(i + 4) + (j + 2) * ldc], _mm256_add_pd(c21, _mm256_loadu_pd(&c[(i + 4) + (j + 2) * ldc])));

            _mm256_storeu_pd(&c[(i + 0) + (j + 3) * ldc], _mm256_add_pd(c30, _mm256_loadu_pd(&c[(i + 0) + (j + 3) * ldc])));
            _mm256_storeu_pd(&c[(i + 4) + (j + 3) * ldc], _mm256_add_pd(c31, _mm256_loadu_pd(&c[(i + 4) + (j + 3) * ldc])));
        }
    }

    if (m8 == m0 && n4 == n0) return;

    double *aa, *bb, *cc;

    // case 1: m4 != m0 
    aa = a + m8; bb = b; cc = c + m8;
    if (m8 != m0) edge_block(m0 - m8, n0, l0, lda, ldb, ldc, aa, bb, cc);

    // case 2: n4 != n0
    aa = a; bb = b + n4 * ldb; cc = c + ldc * n4;
    if (n4 != n0) edge_block(m0, n0 - n4, l0, lda, ldb, ldc, aa, bb, cc);
}
