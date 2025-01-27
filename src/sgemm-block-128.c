#define BLOCK_SIZE 128
#define min(a, b) ((a) < (b) ? (a) : (b))

static void block(int m, int n, int l, int lda, int ldb, int ldc, float* a, float* b, float* c)
{
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < l; k++) {
            float bkj = b[k + j * ldb];
            for (int i = 0; i < m; i++) {
                c[i + j * ldc] += a[i + k * lda] * bkj;
            }
        }
    }
}

void sgemm(int m, int n, int l, float* a, float* b, float* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    int j = 0;
    int k = 0;
    int i = 0;

    for (i = 0; i < m; i += BLOCK_SIZE) {
        int mm = min(BLOCK_SIZE, m - i);
        for (j = 0; j < n; j += BLOCK_SIZE) {
            int nn = min(BLOCK_SIZE, n - j);
            for (k = 0; k < l; k += BLOCK_SIZE) {
                int ll = min(BLOCK_SIZE, l - k);
                block(
                    mm, nn, ll, 
                    lda, ldb, ldc, 
                    a + i + k * lda, 
                    b + k + j * ldb, 
                    c + i + j * ldc
                );
            }
        }
    }
}   