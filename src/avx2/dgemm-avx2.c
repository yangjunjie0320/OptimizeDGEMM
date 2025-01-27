#include "immintrin.h"

void pack(int m, int n, int l, int lda, int ldb, int ldc, double* a, double* b, double* c) {
    // int i, j, k;

    // for (i = 0; i < m; i++) {
    //     for (j = 0; j < n; j++) {
    //         double cij = c[i + j * ldc];
    //         for (k = 0; k < l; k++) {
    //             cij += a[i + k * lda] * b[k + j * ldb];
    //         }
    //         c[i + j * ldc] = cij;
    //     }
    // }
    (void *) a;
    (void *) b;
    (void *) c;
}

void dgemm(int m, int n, int l, double* a, double* b, double* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    int i, j, k;
    int m4 = m & -4;
    int n4 = n & -4;

    __m256d valpha = _mm256_set1_pd(1.0);

    for (i = 0; i < m4; i += 4) {
        for (j = 0; j < n4; j += 4) {
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();

            for (k = 0; k < l; k++) {
                __m256d a0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&a[i + k * lda]));
                __m256d b0 = _mm256_loadu_pd(&b[k + (j + 0) * ldb]);
                __m256d b1 = _mm256_loadu_pd(&b[k + (j + 1) * ldb]);
                __m256d b2 = _mm256_loadu_pd(&b[k + (j + 2) * ldb]);
                __m256d b3 = _mm256_loadu_pd(&b[k + (j + 3) * ldb]);
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

    if (m4 == m && n4 == n) return;
    if (m4 != m) pack(m - m4, n, l, lda, ldb, ldc, a + m4, b, c + m4);
    if (n4 != n) pack(m, n - n4, l, lda, ldb, ldc, a, b + n4 * ldb, c + ldc * n4);
}
