// #include <cstdio>
#include <immintrin.h>

#define min( i, j ) ( (i)<(j) ? (i): (j) )
#define BLOCK_SIZE_M 8
#define BLOCK_SIZE_N 4
#define UNROLL 4

void packing_a(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>3;count_first+=4,count_sub-=4){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm256_store_pd(todst,_mm256_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=4;
        }
    }
    for (;count_sub>0;count_first++,count_sub--){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            *todst=*tosrc;
            tosrc+=leading_dim;
            todst++;
        }
    }
}

void packing_b(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_second=0;count_second<dim_second;count_second+=4){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
        }
    }
}

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


#define KERNEL_K1_8x4_avx2_intrinsics_packing\
    a0 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a));\
    a1 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a+4));\
    b0 = _mm256_broadcast_sd(ptr_packing_b);\
    b1 = _mm256_broadcast_sd(ptr_packing_b+1);\
    b2 = _mm256_broadcast_sd(ptr_packing_b+2);\
    b3 = _mm256_broadcast_sd(ptr_packing_b+3);\
    c00 = _mm256_fmadd_pd(a0,b0,c00);\
    c01 = _mm256_fmadd_pd(a1,b0,c01);\
    c10 = _mm256_fmadd_pd(a0,b1,c10);\
    c11 = _mm256_fmadd_pd(a1,b1,c11);\
    c20 = _mm256_fmadd_pd(a0,b2,c20);\
    c21 = _mm256_fmadd_pd(a1,b2,c21);\
    c30 = _mm256_fmadd_pd(a0,b3,c30);\
    c31 = _mm256_fmadd_pd(a1,b3,c31);\
    ptr_packing_a+=8;ptr_packing_b+=4;k++;
#define macro_kernel_8xkx4_packing\
    c00 = _mm256_setzero_pd();\
    c01 = _mm256_setzero_pd();\
    c10 = _mm256_setzero_pd();\
    c11 = _mm256_setzero_pd();\
    c20 = _mm256_setzero_pd();\
    c21 = _mm256_setzero_pd();\
    c30 = _mm256_setzero_pd();\
    c31 = _mm256_setzero_pd();\
    for (k=0;k<K4;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    for (k=K4;k<K;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    _mm256_storeu_pd(c + (i + 0) * ldc + (j + 0), _mm256_add_pd(c00,_mm256_loadu_pd(c + (i + 0) * ldc + (j + 0))));\
    _mm256_storeu_pd(c + (i + 4) * ldc + (j + 0), _mm256_add_pd(c01,_mm256_loadu_pd(c + (i + 4) * ldc + (j + 0))));\
    _mm256_storeu_pd(c + (i + 0) * ldc + (j + 1), _mm256_add_pd(c10,_mm256_loadu_pd(c + (i + 0) * ldc + (j + 1))));\
    _mm256_storeu_pd(c + (i + 4) * ldc + (j + 1), _mm256_add_pd(c11,_mm256_loadu_pd(c + (i + 4) * ldc + (j + 1))));\
    _mm256_storeu_pd(c + (i + 0) * ldc + (j + 2), _mm256_add_pd(c20,_mm256_loadu_pd(c + (i + 0) * ldc + (j + 2))));\
    _mm256_storeu_pd(c + (i + 4) * ldc + (j + 2), _mm256_add_pd(c21,_mm256_loadu_pd(c + (i + 4) * ldc + (j + 2))));\
    _mm256_storeu_pd(c + (i + 0) * ldc + (j + 3), _mm256_add_pd(c30,_mm256_loadu_pd(c + (i + 0) * ldc + (j + 3))));\
    _mm256_storeu_pd(c + (i + 4) * ldc + (j + 3), _mm256_add_pd(c31,_mm256_loadu_pd(c + (i + 4) * ldc + (j + 3))));
#define KERNEL_K1_4x4_avx2_intrinsics_packing\
    a0 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a));\
    b0 = _mm256_broadcast_sd(ptr_packing_b);\
    b1 = _mm256_broadcast_sd(ptr_packing_b+1);\
    b2 = _mm256_broadcast_sd(ptr_packing_b+2);\
    b3 = _mm256_broadcast_sd(ptr_packing_b+3);\
    c00 = _mm256_fmadd_pd(a0,b0,c00);\
    c10 = _mm256_fmadd_pd(a0,b1,c10);\
    c20 = _mm256_fmadd_pd(a0,b2,c20);\
    c30 = _mm256_fmadd_pd(a0,b3,c30);\
    ptr_packing_a+=4;ptr_packing_b+=4;k++;
#define macro_kernel_4xkx4_packing\
    c00 = _mm256_setzero_pd();\
    c10 = _mm256_setzero_pd();\
    c20 = _mm256_setzero_pd();\
    c30 = _mm256_setzero_pd();\
    for (k=0;k<K4;){\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
    }\
    for (k=K4;k<K;){\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
    }\
    _mm256_storeu_pd(c + i * ldc + (j + 0), _mm256_add_pd(c00,_mm256_loadu_pd(c + i * ldc + (j + 0))));\
    _mm256_storeu_pd(c + i * ldc + (j + 1), _mm256_add_pd(c10,_mm256_loadu_pd(c + i * ldc + (j + 1))));\
    _mm256_storeu_pd(c + i * ldc + (j + 2), _mm256_add_pd(c20,_mm256_loadu_pd(c + i * ldc + (j + 2))));\
    _mm256_storeu_pd(c + i * ldc + (j + 3), _mm256_add_pd(c30,_mm256_loadu_pd(c + i * ldc + (j + 3))));

void macro_kernel(int M, int N, int K, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    int M8=M&-8,N4=N&-4,K4=K&-4;
    double *ptr_packing_a = A;
    double *ptr_packing_b = B;

    double* c = C;
    int ldc = LDC;

    double alpha = 1.0;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,a0,a1,b0,b1,b2,b3;
    __m256d c00,c01,c10,c11,c20,c21,c30,c31;
    __m256d c0,c1,c2,c3;
    for (i=0;i<M8;i+=8){
        for (j=0;j<N4;j+=4){
            ptr_packing_a=A+i*K;ptr_packing_b=B+j*K;
            macro_kernel_8xkx4_packing
        }
    }
    for (i=M8;i<M;i+=4){
        for (j=0;j<N4;j+=4){
            ptr_packing_a=A+i*K;ptr_packing_b=B+j*K;
            macro_kernel_4xkx4_packing
        }
    }
}

#define M_BLOCKING 192
#define N_BLOCKING 2048
#define K_BLOCKING 384

void dgemm(int m, int n, int l, double* a, double* b, double* c) {
    int m0 = m, n0 = n, l0 = l;
    int m1 = 0, n1 = 0, l1 = 0;
    int ii, jj, kk;

    int lda = m;
    int ldb = l;  
    int ldc = m;

    double* a_buffer = (double*) aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(double));
    double* b_buffer = (double*) aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(double));

    // n, l, m
    for (jj = 0; jj < n0;) {
        n1 = min(N_BLOCKING, n0 - jj);
        for (kk = 0; kk < l0;) {
            l1 = min(K_BLOCKING, l0 - kk);
            packing_b(b + kk + jj * ldb, b_buffer, ldb, l1, n1);
            for (ii = 0; ii < m0; ) {
                m1 = min(M_BLOCKING, m0 - ii);
                packing_a(a + ii + kk * lda, a_buffer, lda, l1, m1);
                macro_kernel(
                    m1, n1, l1,
                    a_buffer, lda,
                    b_buffer, ldb,
                    c + ii + jj * ldc, ldc
                );
                ii += m1;
            }
            kk += l1;
        }
        jj += n1;
    }

    free(a_buffer);
    free(b_buffer);
}