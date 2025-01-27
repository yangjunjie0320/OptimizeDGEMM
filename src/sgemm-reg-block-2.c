#define BLOCK_SIZE_M 2
#define BLOCK_SIZE_N 2
#define BLOCK_SIZE_L 64

#define min(a, b) ((a) < (b) ? (a) : (b))

static void block(int m, int n, int l, int lda, int ldb, int ldc, float* a, float* b, float* c)
{   
    int p;
    register float
        c_reg_00, c_reg_01,
        c_reg_10, c_reg_11;

    register float a_reg_00, a_reg_01;

    c_reg_00 = 0.0f;
    c_reg_01 = 0.0f;

    c_reg_10 = 0.0f;
    c_reg_11 = 0.0f;

    for (p = 0; p < l; p++) {
        a_reg_00 = a[0 + p * lda];
        a_reg_01 = a[1 + p * lda];

        c_reg_00 += a_reg_00 * b[p + 0 * ldb];
        c_reg_01 += a_reg_01 * b[p + 0 * ldb];

        c_reg_10 += a_reg_00 * b[p + 1 * ldb];
        c_reg_11 += a_reg_01 * b[p + 1 * ldb];
    }

    c[0 + 0 * ldc] += c_reg_00;
    c[1 + 0 * ldc] += c_reg_01;

    c[0 + 1 * ldc] += c_reg_10;
    c[1 + 1 * ldc] += c_reg_11;
}

void sgemm(int m, int n, int l, float* a, float* b, float* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    for (int i = 0; i < m; i += BLOCK_SIZE_M) {
        int mm = min(BLOCK_SIZE_M, m - i);

        for (int j = 0; j < n; j += BLOCK_SIZE_N) {
            int nn = min(BLOCK_SIZE_N, n - j);
        
            for (int k = 0; k < l; k += BLOCK_SIZE_L) {
                int ll = min(BLOCK_SIZE_L, l - k);
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
