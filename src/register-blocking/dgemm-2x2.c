#define BLOCK_SIZE 2

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
    int m2 = m0 & -BLOCK_SIZE;
    int n2 = n0 & -BLOCK_SIZE;

    register double c00, c01, c10, c11;

    for (i = 0; i < m2; i += BLOCK_SIZE) {
        for (j = 0; j < n2; j += BLOCK_SIZE) {
            c00 = c[(i + 0) + (j + 0) * ldc];
            c01 = c[(i + 0) + (j + 1) * ldc];
            c10 = c[(i + 1) + (j + 0) * ldc];
            c11 = c[(i + 1) + (j + 1) * ldc];

            for (k = 0; k < l; k++) {
                double a0k = a[(i + 0) + k * lda];
                double a1k = a[(i + 1) + k * lda];

                c00 += a0k * b[k + (j + 0) * ldb];
                c01 += a0k * b[k + (j + 1) * ldb];
                c10 += a1k * b[k + (j + 0) * ldb];
                c11 += a1k * b[k + (j + 1) * ldb];
            }

            c[(i + 0) + (j + 0) * ldc] = c00;
            c[(i + 0) + (j + 1) * ldc] = c01;
            c[(i + 1) + (j + 0) * ldc] = c10;
            c[(i + 1) + (j + 1) * ldc] = c11;
        }
    }
    
    if (m2 == m0 && n2 == n0) return;

    double *aa, *bb, *cc;

    // case 1: m2 != m0 
    aa = a + m2; bb = b; cc = c + m2;
    if (m2 != m0) edge_block(m0 - m2, n0, l0, lda, ldb, ldc, aa, bb, cc);

    // case 2: n2 != n0
    aa = a; bb = b + n2 * ldb; cc = c + ldc * n2;
    if (n2 != n0) edge_block(m0, n0 - n2, l0, lda, ldb, ldc, aa, bb, cc);
}
