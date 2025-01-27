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
    int m4 = m & -4;
    int n4 = n & -4;

    register double c00, c01, c10, c11;
    register double c20, c21, c30, c31;
    register double c02, c03, c12, c13;
    register double c22, c23, c32, c33;

    for (i = 0; i < m4; i += 4) {
        for (j = 0; j < n4; j += 4) {
            c00 = c[(i + 0) + (j + 0) * ldc];
            c01 = c[(i + 0) + (j + 1) * ldc];
            c10 = c[(i + 1) + (j + 0) * ldc];
            c11 = c[(i + 1) + (j + 1) * ldc];

            c20 = c[(i + 2) + (j + 0) * ldc];
            c21 = c[(i + 2) + (j + 1) * ldc];
            c30 = c[(i + 3) + (j + 0) * ldc];
            c31 = c[(i + 3) + (j + 1) * ldc];

            c02 = c[(i + 0) + (j + 2) * ldc];
            c03 = c[(i + 0) + (j + 3) * ldc];
            c12 = c[(i + 1) + (j + 2) * ldc];
            c13 = c[(i + 1) + (j + 3) * ldc];

            c22 = c[(i + 2) + (j + 2) * ldc];
            c23 = c[(i + 2) + (j + 3) * ldc];
            c32 = c[(i + 3) + (j + 2) * ldc];
            c33 = c[(i + 3) + (j + 3) * ldc];

            for (k = 0; k < l; k++) {
                // wish to be stored on cache1
                double a0k = a[(i + 0) + k * lda];
                double a1k = a[(i + 1) + k * lda];
                double a2k = a[(i + 2) + k * lda];
                double a3k = a[(i + 3) + k * lda];

                c00 += a0k * b[k + (j + 0) * ldb];
                c01 += a0k * b[k + (j + 1) * ldb];
                c02 += a0k * b[k + (j + 2) * ldb];
                c03 += a0k * b[k + (j + 3) * ldb];

                c10 += a1k * b[k + (j + 0) * ldb];
                c11 += a1k * b[k + (j + 1) * ldb];
                c12 += a1k * b[k + (j + 2) * ldb];
                c13 += a1k * b[k + (j + 3) * ldb];

                c20 += a2k * b[k + (j + 0) * ldb];
                c21 += a2k * b[k + (j + 1) * ldb];
                c22 += a2k * b[k + (j + 2) * ldb];
                c23 += a2k * b[k + (j + 3) * ldb];

                c30 += a3k * b[k + (j + 0) * ldb];
                c31 += a3k * b[k + (j + 1) * ldb];
                c32 += a3k * b[k + (j + 2) * ldb];
                c33 += a3k * b[k + (j + 3) * ldb];
            }

            c[(i + 0) + (j + 0) * ldc] = c00;
            c[(i + 0) + (j + 1) * ldc] = c01;
            c[(i + 0) + (j + 2) * ldc] = c02;
            c[(i + 0) + (j + 3) * ldc] = c03;

            c[(i + 1) + (j + 0) * ldc] = c10;
            c[(i + 1) + (j + 1) * ldc] = c11;
            c[(i + 1) + (j + 2) * ldc] = c12;
            c[(i + 1) + (j + 3) * ldc] = c13;

            c[(i + 2) + (j + 0) * ldc] = c20;
            c[(i + 2) + (j + 1) * ldc] = c21;
            c[(i + 2) + (j + 2) * ldc] = c22;
            c[(i + 2) + (j + 3) * ldc] = c23;

            c[(i + 3) + (j + 0) * ldc] = c30;
            c[(i + 3) + (j + 1) * ldc] = c31;
            c[(i + 3) + (j + 2) * ldc] = c32;
            c[(i + 3) + (j + 3) * ldc] = c33;
        }
    }

    if (m4 == m && n4 == n) return;
    if (m4 != m) pack(m - m4, n, l, lda, ldb, ldc, a + m4, b, c + m4);
    if (n4 != n) pack(m, n - n4, l, lda, ldb, ldc, a, b + n4, c);
}
