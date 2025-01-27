void dgemm(int m, int n, int l, double* a, double* b, double* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    for (int j = 0; j < n; j++) {
        for (int k = 0; k < l; k++) {
            double bkj = b[k + j * ldb];
            for (int i = 0; i < m; i++) {
                c[i + j * ldc] += a[i + k * lda] * bkj;
            }
        }
    }
}   