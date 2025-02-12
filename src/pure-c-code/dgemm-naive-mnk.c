void dgemm(int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    // A is M x K
    // B is K x N
    // C is M x N
    int LDA = M;
    int LDB = K;
    int LDC = M;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double Cmn = 0.0;
            for (int k = 0; k < K; k++) {
                Cmn += A[m + k * LDA] * B[k + n * LDB];
            }
            C[m + n * LDC] += Cmn;
        }
    }
}   
