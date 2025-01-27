#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DoubleMatrix;

typedef std::chrono::high_resolution_clock::time_point TimePoint;
double calculate_time_difference(TimePoint t1, TimePoint t2) {
    auto s = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    return static_cast<double>(s);
}

extern "C" {
    void dgemm(int, int, int, double*, double*, double*);
    void dgemm_blas(int, int, int, double*, double*, double*);
}

std::tuple<DoubleMatrix, double> mm_sol(const DoubleMatrix& ma, const DoubleMatrix& mb)
{
    DoubleMatrix mc(ma.rows(), mb.cols());
    mc.setZero(); // optional, since we specify beta=0 anyway

    double* pa = (double*) ma.data();
    double* pb = (double*) mb.data();
    double* pc = (double*) mc.data();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    dgemm(ma.rows(), ma.cols(), mb.cols(), pa, pb, pc);
    // mc += ma * mb;
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::make_tuple(mc, calculate_time_difference(t1, t2));
}

std::tuple<DoubleMatrix, double> mm_ref(const DoubleMatrix &ma, const DoubleMatrix &mb)
{
    DoubleMatrix mc(ma.rows(), mb.cols());
    mc.setZero(); // optional, since we specify beta=0 anyway

    double* pa = (double*) ma.data();
    double* pb = (double*) mb.data();
    double* pc = (double*) mc.data();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    dgemm_blas(ma.rows(), ma.cols(), mb.cols(), pa, pb, pc);
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::make_tuple(mc, calculate_time_difference(t1, t2));
}

std::vector<double> gflops(const std::vector<double> &tt, const int s) {
    std::vector<double> gg;
    for (auto t : tt) {
        gg.push_back(2.0 * s * s * s / t / 1e9);
    }
    return gg;
}

double average(std::vector<double> &xx) {
    double x0 = 0.0;
    for (auto x : xx) {
        x0 += x;
    }
    return x0 / xx.size();
}

double deviation(std::vector<double> &xx) {
    assert(xx.size() > 1);
    int d = xx.size() - 1;

    double x0 = 0.0;
    for (auto x : xx) {
        x0 += x;
    }
    x0 = x0 / xx.size();

    double n = 0.0;
    for (auto x : xx) {
        n += (x - x0) * (x - x0);
    }
    return std::sqrt(n / d);
}

int main(int argc, char* argv[]) {
    std::vector<double> tt0;
    std::vector<double> tt1;

    int L = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    assert(L % 4 == 0);
    printf("L = %d, N = %d\n", L, N);

    double eps = std::numeric_limits<double>::epsilon();

    for (int i = 0; i < N; i++) {
        DoubleMatrix A;
        A.resize(L, L);
        A.setRandom();

        DoubleMatrix B;
        B.resize(L, L);
        B.setRandom();

        auto [C0, dt0] = mm_ref(A, B);
        auto [C1, dt1] = mm_sol(A, B);

        tt0.push_back(dt0);
        tt1.push_back(dt1);
        
        auto err = (C0 - C1).array().abs().maxCoeff();
        assert(err < 1e-10);
    }

    std::sort(tt0.begin(), tt0.end());
    tt0.erase(tt0.begin());
    tt0.erase(tt0.end() - 1);

    auto gg0 = gflops(tt0, L);
    // assert (deviation(gg0) / average(gg0) < 0.2);
    printf("MM_REF t = %6.2e sec, GFLOPS = %6.2f\n", average(tt0), average(gg0));

    std::sort(tt1.begin(), tt1.end());
    tt1.erase(tt1.begin());
    tt1.erase(tt1.end() - 1);

    auto gg1 = gflops(tt1, L);
    // assert (deviation(gg1) / average(gg1) < 0.1);
    printf("MM_SOL t = %6.2e sec, GFLOPS = %6.2f\n", average(tt1), average(gg1));

    return 0;
}
