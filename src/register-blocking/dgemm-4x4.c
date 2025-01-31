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

    int m0 = m, n0 = n, l0 = l;
    int m4 = m0 & -BLOCK_SIZE;
    int n4 = n0 & -BLOCK_SIZE;

    register double c00, c01, c10, c11;
    register double c20, c21, c30, c31;
    register double c02, c03, c12, c13;
    register double c22, c23, c32, c33;

    for (i = 0; i < m4; i += BLOCK_SIZE) {
        for (j = 0; j < n4; j += BLOCK_SIZE) {
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

    if (m4 == m0 && n4 == n0) return;

    double *aa, *bb, *cc;

    // case 1: m4 != m0 
    aa = a + m4; bb = b; cc = c + m4;
    if (m4 != m0) edge_block(m0 - m4, n0, l0, lda, ldb, ldc, aa, bb, cc);

    // case 2: n4 != n0
    aa = a; bb = b + n4 * ldb; cc = c + ldc * n4;
    if (n4 != n0) edge_block(m0, n0 - n4, l0, lda, ldb, ldc, aa, bb, cc);
}

