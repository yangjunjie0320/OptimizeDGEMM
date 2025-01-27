#define BLOCK_SIZE_M 2
#define BLOCK_SIZE_N 2
#define BLOCK_SIZE_L 64

#define min(a, b) ((a) < (b) ? (a) : (b))

static void block(int m, int n, int l, int lda, int ldb, int ldc, float* a, float* b, float* c)
{   
    int p;
    register float
    // volatile float
        c_reg_00, c_reg_01, // c_reg_02, c_reg_03,
        c_reg_10, c_reg_11; // c_reg_12, c_reg_13,
        // c_reg_20, c_reg_21, // c_reg_22, c_reg_23,
        // c_reg_30, c_reg_31, // c_reg_32, c_reg_33;

    c_reg_00 = 0.0f;
    c_reg_01 = 0.0f;
    // c_reg_02 = 0.0f;
    // c_reg_03 = 0.0f;

    c_reg_10 = 0.0f;
    c_reg_11 = 0.0f;
    // c_reg_12 = 0.0f;
    // c_reg_13 = 0.0f;

    // c_reg_20 = 0.0f;
    // c_reg_21 = 0.0f;
    // c_reg_22 = 0.0f;
    // c_reg_23 = 0.0f;

    // register float
    // // volatile float 
    //     a_reg_00, a_reg_01, a_reg_02, a_reg_03;

    for (p = 0; p < l; p++) {
        float a_reg_00, a_reg_01;
        a_reg_00 = a[0 + p * lda];
        a_reg_01 = a[1 + p * lda];
        // a_reg_02 = a[2 + p * lda];
        // a_reg_03 = a[3 + p * lda];

        c_reg_00 += a_reg_00 * b[p + 0 * ldb];
        c_reg_01 += a_reg_01 * b[p + 0 * ldb];
        // c_reg_02 += a_reg_02 * b[p + 0 * ldb];
        // c_reg_03 += a_reg_03 * b[p + 0 * ldb];

        c_reg_10 += a_reg_00 * b[p + 1 * ldb];
        c_reg_11 += a_reg_01 * b[p + 1 * ldb];
        // c_reg_12 += a_reg_02 * b[p + 1 * ldb];
        // c_reg_13 += a_reg_03 * b[p + 1 * ldb];

        // c_reg_20 += a_reg_00 * b[p + 2 * ldb];
        // c_reg_21 += a_reg_01 * b[p + 2 * ldb];
        // c_reg_22 += a_reg_02 * b[p + 2 * ldb];
        // c_reg_23 += a_reg_03 * b[p + 2 * ldb];

        // c_reg_30 += a_reg_00 * b[p + 3 * ldb];
        // c_reg_31 += a_reg_01 * b[p + 3 * ldb];
        // c_reg_32 += a_reg_02 * b[p + 3 * ldb];
        // c_reg_33 += a_reg_03 * b[p + 3 * ldb];
    }

    c[0 + 0 * ldc] += c_reg_00;
    c[1 + 0 * ldc] += c_reg_01;
    // c[2 + 0 * ldc] += c_reg_02;
    // c[3 + 0 * ldc] += c_reg_03;

    c[0 + 1 * ldc] += c_reg_10;
    c[1 + 1 * ldc] += c_reg_11;
    // c[2 + 1 * ldc] += c_reg_12;
    // c[3 + 1 * ldc] += c_reg_13;

    // c[0 + 2 * ldc] += c_reg_20;
    // c[1 + 2 * ldc] += c_reg_21;
    // c[2 + 2 * ldc] += c_reg_22;
    // c[3 + 2 * ldc] += c_reg_23;

    // c[0 + 3 * ldc] += c_reg_30;
    // c[1 + 3 * ldc] += c_reg_31;
    // c[2 + 3 * ldc] += c_reg_32;
    // c[3 + 3 * ldc] += c_reg_33;
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
