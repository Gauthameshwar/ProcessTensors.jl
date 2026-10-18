// Replay one column-major ComplexF64 matrix with LAPACKE_zgesvd (thin vectors).
// Links the same LP64 OpenBLAS as C++ ACE. Does not use Julia's ILP64 library.
#include <lapacke.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <time.h>

static double elapsed_s(const timespec &a, const timespec &b)
{
    return (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec);
}

static int run_once(
    lapack_int m, lapack_int n, const std::vector<std::complex<double>> &source,
    std::vector<double> &sigma, double &lwork_out)
{
    const lapack_int lda = m;
    const lapack_int minmn = std::min(m, n);
    std::vector<std::complex<double>> a = source;
    std::vector<std::complex<double>> u(static_cast<size_t>(m) * static_cast<size_t>(minmn));
    std::vector<std::complex<double>> vt(static_cast<size_t>(minmn) * static_cast<size_t>(n));
    sigma.assign(static_cast<size_t>(minmn), 0.0);
    std::vector<double> superb(static_cast<size_t>(std::max(minmn - 1, lapack_int{1})));
    std::vector<double> rwork(static_cast<size_t>(5 * std::max(minmn, lapack_int{1})));
    std::vector<std::complex<double>> work(1);
    lapack_int lwork = -1;
    lapack_int info = LAPACKE_zgesvd_work(
        LAPACK_COL_MAJOR, 'S', 'S', m, n, reinterpret_cast<lapack_complex_double *>(a.data()), lda,
        sigma.data(), reinterpret_cast<lapack_complex_double *>(u.data()), m,
        reinterpret_cast<lapack_complex_double *>(vt.data()), minmn,
        reinterpret_cast<lapack_complex_double *>(work.data()), lwork, rwork.data());
    if (info != 0)
        return info;
    lwork_out = work[0].real();
    lwork = static_cast<lapack_int>(lwork_out);
    if (lwork < 1)
        lwork = 1;
    work.assign(static_cast<size_t>(lwork), std::complex<double>(0.0, 0.0));
    a = source;
    info = LAPACKE_zgesvd_work(
        LAPACK_COL_MAJOR, 'S', 'S', m, n, reinterpret_cast<lapack_complex_double *>(a.data()), lda,
        sigma.data(), reinterpret_cast<lapack_complex_double *>(u.data()), m,
        reinterpret_cast<lapack_complex_double *>(vt.data()), minmn,
        reinterpret_cast<lapack_complex_double *>(work.data()), lwork, rwork.data());
    if (info != 0)
        return info;
    double num = 0.0;
    double den = 0.0;
    for (lapack_int col = 0; col < n; ++col) {
        for (lapack_int row = 0; row < m; ++row) {
            std::complex<double> acc = 0.0;
            for (lapack_int k = 0; k < minmn; ++k) {
                acc += u[static_cast<size_t>(k) * m + row] * sigma[static_cast<size_t>(k)] *
                       vt[static_cast<size_t>(col) * minmn + k];
            }
            const std::complex<double> diff = source[static_cast<size_t>(col) * m + row] - acc;
            num += std::norm(diff);
            den += std::norm(source[static_cast<size_t>(col) * m + row]);
        }
    }
    const double rel = den == 0.0 ? 0.0 : std::sqrt(num / den);
    std::cout << "recon " << rel << "\n";
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "usage: gesvd_replay matrix.bin repeats label\n";
        return 2;
    }
    const std::string path = argv[1];
    const int repeats = std::max(1, std::atoi(argv[2]));
    const std::string label = argv[3];
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return 2;
    }
    int64_t m64 = 0, n64 = 0;
    in.read(reinterpret_cast<char *>(&m64), sizeof(m64));
    in.read(reinterpret_cast<char *>(&n64), sizeof(n64));
    if (!in || m64 < 1 || n64 < 1) {
        std::cerr << "bad matrix header\n";
        return 2;
    }
    const auto count = static_cast<size_t>(m64) * static_cast<size_t>(n64);
    std::vector<std::complex<double>> source(count);
    in.read(reinterpret_cast<char *>(source.data()), static_cast<std::streamsize>(count * sizeof(std::complex<double>)));
    if (!in) {
        std::cerr << "truncated matrix payload\n";
        return 2;
    }
    const lapack_int m = static_cast<lapack_int>(m64);
    const lapack_int n = static_cast<lapack_int>(n64);
    std::vector<double> sigma;
    double lwork = 0.0;
    if (run_once(m, n, source, sigma, lwork) != 0) {
        std::cerr << "LAPACKE_zgesvd failed\n";
        return 1;
    }
    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int rep = 0; rep < repeats; ++rep) {
        std::vector<std::complex<double>> a = source;
        const lapack_int minmn = std::min(m, n);
        std::vector<std::complex<double>> u(static_cast<size_t>(m) * static_cast<size_t>(minmn));
        std::vector<std::complex<double>> vt(static_cast<size_t>(minmn) * static_cast<size_t>(n));
        std::vector<double> s(static_cast<size_t>(minmn));
        std::vector<double> superb(static_cast<size_t>(std::max(minmn - 1, lapack_int{1})));
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const lapack_int info = LAPACKE_zgesvd(
            LAPACK_COL_MAJOR, 'S', 'S', m, n, reinterpret_cast<lapack_complex_double *>(a.data()), m,
            s.data(), reinterpret_cast<lapack_complex_double *>(u.data()), m,
            reinterpret_cast<lapack_complex_double *>(vt.data()), minmn, superb.data());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (info != 0) {
            std::cerr << "timed gesvd failed\n";
            return 1;
        }
        times.push_back(elapsed_s(t0, t1));
        sigma.swap(s);
    }
    std::sort(times.begin(), times.end());
    const double tmin = times.front();
    const double tmax = times.back();
    const double tmed = times[times.size() / 2];
    const std::string sv_path = std::string(argv[1]) + ".lapacke_sigma.txt";
    std::ofstream sv(sv_path);
    for (double value : sigma)
        sv << value << "\n";
    std::cout << "label " << label << "\n";
    std::cout << "m " << m64 << "\n";
    std::cout << "n " << n64 << "\n";
    std::cout << "lda " << m64 << "\n";
    std::cout << "lwork " << lwork << "\n";
    std::cout << "min_s " << tmin << "\n";
    std::cout << "median_s " << tmed << "\n";
    std::cout << "max_s " << tmax << "\n";
    std::cout << "samples " << repeats << "\n";
    std::cout << "job SS\n";
    std::cout << "abi LP64\n";
    return 0;
}
