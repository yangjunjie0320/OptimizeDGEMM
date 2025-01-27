void pack(int m, int n, int l, int lda, int ldb, int ldc, double* a, double* b, double* c) {
    int i, j, k;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double cij = c[i + j * ldc];
            for (k = 0; k < l; k++) {
                cij += a[i + k * lda] * b[k + j * ldb];
            }
            c[i + j * ldc] = cij;
        }
    }
}

void dgemm(int m, int n, int l, double* a, double* b, double* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    int i, j, k;
    int m2 = m & -2;
    int n2 = n & -2;

    register double c00, c01, c10, c11;
    register double a0k, a1k, bk0, bk1;

    for (i = 0; i < m2; i += 2) {
        for (j = 0; j < n2; j += 2) {
            c00 = c[(i + 0) + (j + 0) * ldc];
            c01 = c[(i + 0) + (j + 1) * ldc];
            c10 = c[(i + 1) + (j + 0) * ldc];
            c11 = c[(i + 1) + (j + 1) * ldc];

            for (k = 0; k < l; k++) {
                a0k = a[(i + 0) + k * lda];
                a1k = a[(i + 1) + k * lda];

                bk0 = b[k + (j + 0) * ldb];
                bk1 = b[k + (j + 1) * ldb];

                c00 += a0k * bk0;
                c01 += a0k * bk1;
                c10 += a1k * bk0;
                c11 += a1k * bk1;
            }

            c[(i + 0) + (j + 0) * ldc] = c00;
            c[(i + 0) + (j + 1) * ldc] = c01;
            c[(i + 1) + (j + 0) * ldc] = c10;
            c[(i + 1) + (j + 1) * ldc] = c11;
        }
    }

    if (m2 == m && n2 == n) return;
    if (m2 != m) pack(m - m2, n, l, lda, ldb, ldc, a + m2, b, c + m2);
    if (n2 != n) pack(m, n - n2, l, lda, ldb, ldc, a, b + n2, c);
}
