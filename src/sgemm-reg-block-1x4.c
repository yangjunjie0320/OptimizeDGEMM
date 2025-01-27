#define BLOCK_SIZE 4
#define min(a, b) ((a) < (b) ? (a) : (b))

static void block_1x4(int m, int n, int l, int lda, int ldb, int ldc, float* a, float* b, float* c)
{   
    register float
        c_reg_00, c_reg_01, c_reg_02, c_reg_03;

    c_reg_00 = 0.0f;
    c_reg_01 = 0.0f;
    c_reg_02 = 0.0f;
    c_reg_03 = 0.0f;

    register float a_reg_0k;

    for (int k = 0; k < l; k += 1) {
        a_reg_0k = a[0 + k * lda];
        c_reg_00 += a_reg_0k * b[k + 0 * ldb];
        c_reg_01 += a_reg_0k * b[k + 1 * ldb];
        c_reg_02 += a_reg_0k * b[k + 2 * ldb];
        c_reg_03 += a_reg_0k * b[k + 3 * ldb];
    }

    c[0 + 0 * ldc] += c_reg_00;
    c[0 + 1 * ldc] += c_reg_01;
    c[0 + 2 * ldc] += c_reg_02;
    c[0 + 3 * ldc] += c_reg_03;
}

void sgemm(int m, int n, int l, float* a, float* b, float* c)
{
    int lda = l;
    int ldb = n;
    int ldc = n;

    for (int i = 0; i < m; i += 1) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            block_1x4(
                1, BLOCK_SIZE, l, 
                lda, ldb, ldc, 
                a + i + 0 * lda, 
                b + 0 + j * ldb, 
                c + i + j * ldc
            );
        }
    }   
}