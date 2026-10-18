/* Count and time LAPACK SVD kernels without changing their numerics.
 *
 * Julia 1.12 ILP64 OpenBLAS exports zgesvd_64_ / zgesdd_64_ (64-bit integers).
 * C++ ACE links LP64 OpenBLAS and enters through LAPACKE_zgesvd.
 * Both paths are interposed. svd_kernel_reset/total_ns are the Julia-side hook.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int32_t lp64_int;
typedef int64_t ilp64_int;

static uint64_t g_calls;
static uint64_t g_total_ns;
static uint64_t g_last_ns;
static int64_t g_last_m;
static int64_t g_last_n;
static int64_t g_last_lda;
static int64_t g_last_lwork;
static char g_last_jobu;
static char g_last_jobvt;
static char g_last_kind;
static FILE *g_log;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void open_log(void)
{
    const char *path;
    if (g_log)
        return;
    path = getenv("SVD_COUNT_LOG");
    if (!path || !path[0])
        return;
    g_log = fopen(path, "a");
    if (!g_log)
        return;
    setvbuf(g_log, NULL, _IOLBF, 0);
}

static void note(char kind, char jobu, char jobvt, int64_t m, int64_t n, int64_t lda, int64_t lwork, uint64_t dt)
{
    g_calls += 1;
    g_total_ns += dt;
    g_last_ns = dt;
    g_last_m = m;
    g_last_n = n;
    g_last_lda = lda;
    g_last_lwork = lwork;
    g_last_jobu = jobu;
    g_last_jobvt = jobvt;
    g_last_kind = kind;
    open_log();
    if (g_log) {
        fprintf(g_log, "%c,%c,%c,%ld,%ld,%ld,%ld,%.9f\n",
                kind, jobu, jobvt, (long)m, (long)n, (long)lda, (long)lwork, (double)dt * 1e-9);
    }
}

void svd_kernel_reset(void)
{
    g_calls = 0;
    g_total_ns = 0;
    g_last_ns = 0;
    g_last_m = 0;
    g_last_n = 0;
    g_last_lda = 0;
    g_last_lwork = 0;
    g_last_jobu = '?';
    g_last_jobvt = '?';
    g_last_kind = '?';
}

uint64_t svd_kernel_calls(void) { return g_calls; }
uint64_t svd_kernel_total_ns(void) { return g_total_ns; }
uint64_t svd_kernel_last_ns(void) { return g_last_ns; }
int64_t svd_kernel_last_m(void) { return g_last_m; }
int64_t svd_kernel_last_n(void) { return g_last_n; }
int64_t svd_kernel_last_lda(void) { return g_last_lda; }
int64_t svd_kernel_last_lwork(void) { return g_last_lwork; }
int svd_kernel_last_jobu(void) { return (int)g_last_jobu; }
int svd_kernel_last_jobvt(void) { return (int)g_last_jobvt; }
int svd_kernel_last_kind(void) { return (int)g_last_kind; }

/* Julia ILP64, gfortran hidden string lengths included. */
void zgesvd_64_(
    char *jobu, char *jobvt, ilp64_int *m, ilp64_int *n, void *a, ilp64_int *lda,
    double *s, void *u, ilp64_int *ldu, void *vt, ilp64_int *ldvt, void *work,
    ilp64_int *lwork, double *rwork, ilp64_int *info, long jobu_len, long jobvt_len)
{
    static void (*real_fn)(char *, char *, ilp64_int *, ilp64_int *, void *, ilp64_int *,
                           double *, void *, ilp64_int *, void *, ilp64_int *, void *,
                           ilp64_int *, double *, ilp64_int *, long, long) = NULL;
    uint64_t t0, dt;
    int64_t lwork_in;
    if (!real_fn)
        real_fn = dlsym(RTLD_NEXT, "zgesvd_64_");
    lwork_in = lwork ? (int64_t)(*lwork) : 0;
    t0 = now_ns();
    real_fn(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info, jobu_len, jobvt_len);
    dt = now_ns() - t0;
    note('v', jobu ? *jobu : '?', jobvt ? *jobvt : '?', m ? (int64_t)(*m) : 0, n ? (int64_t)(*n) : 0,
         lda ? (int64_t)(*lda) : 0, lwork_in, dt);
}

void zgesdd_64_(
    char *job, ilp64_int *m, ilp64_int *n, void *a, ilp64_int *lda, double *s, void *u,
    ilp64_int *ldu, void *vt, ilp64_int *ldvt, void *work, ilp64_int *lwork, double *rwork,
    ilp64_int *iwork, ilp64_int *info, long job_len)
{
    static void (*real_fn)(char *, ilp64_int *, ilp64_int *, void *, ilp64_int *, double *,
                           void *, ilp64_int *, void *, ilp64_int *, void *, ilp64_int *,
                           double *, ilp64_int *, ilp64_int *, long) = NULL;
    uint64_t t0, dt;
    int64_t lwork_in;
    if (!real_fn)
        real_fn = dlsym(RTLD_NEXT, "zgesdd_64_");
    lwork_in = lwork ? (int64_t)(*lwork) : 0;
    t0 = now_ns();
    real_fn(job, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info, job_len);
    dt = now_ns() - t0;
    note('d', job ? *job : '?', job ? *job : '?', m ? (int64_t)(*m) : 0, n ? (int64_t)(*n) : 0,
         lda ? (int64_t)(*lda) : 0, lwork_in, dt);
}

/* C++ LP64 LAPACKE entry used by Eigen JacobiSVD. */
typedef int32_t (*lapacke_zgesvd_t)(
    int, char, char, lp64_int, lp64_int, void *, lp64_int, double *, void *, lp64_int, void *,
    lp64_int, double *);

int32_t LAPACKE_zgesvd(
    int matrix_layout, char jobu, char jobvt, lp64_int m, lp64_int n, void *a, lp64_int lda,
    double *s, void *u, lp64_int ldu, void *vt, lp64_int ldvt, double *superb)
{
    static lapacke_zgesvd_t real_fn = NULL;
    uint64_t t0, dt;
    int32_t info;
    if (!real_fn)
        real_fn = (lapacke_zgesvd_t)dlsym(RTLD_NEXT, "LAPACKE_zgesvd");
    t0 = now_ns();
    info = real_fn(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    dt = now_ns() - t0;
    note('L', jobu, jobvt, (int64_t)m, (int64_t)n, (int64_t)lda, -1, dt);
    return info;
}
