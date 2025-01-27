void dgemm(int m, int n, int l, double* a, double* b, double* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double cij = 0.0;
            for (int k = 0; k < l; k++) {
                cij += a[i + k * lda] * b[k + j * ldb];
            }
            c[i + j * ldc] += cij;
        }
    }
}   
