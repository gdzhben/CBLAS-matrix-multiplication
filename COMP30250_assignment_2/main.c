#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>

struct timeval tv1, tv2;
struct timezone tz;

void print_matrix(int n, double* matrix);
double* get_array(int size, double value);
void straightforward_nonblocked_ijk_algorithm(int n, double* a, double* b, double* result);
void blas_blocked_ijk_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result);
void blas_blocked_kij_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result);
void multiply_nonblocked_ijk_algorithm(int n, double* a, double* b,
        double* result);
void multiply_nonblocked_kij_algorithm(int n, double* a, double* b,
        double* result);

int main(int argc, char** argv) {
    int n = 100;
    int i, j, k;

    double a[n * n], b[n * n];
    srand(time(NULL));
    for (i = 0; i < n * n; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    double* result = (double*) calloc(n*n, sizeof (double));

    straightforward_nonblocked_ijk_algorithm(n, a, b, result);
    //print_matrix(n, result);
    for (j = n / 10; j <= n; j = j + 10) {
        blas_blocked_ijk_algorithm(n, j, a, b, result);
        //print_matrix(n, result);
    }

    for (j = n / 10; j <= n; j = j + 10) {
        blas_blocked_kij_algorithm(n, j, a, b, result);
    }

    //print_matrix(n, result);
    multiply_nonblocked_ijk_algorithm(n, a, b, result);
    multiply_nonblocked_kij_algorithm(n, a, b, result);
    //print_matrix(n, result);


    return (EXIT_SUCCESS);
}

void straightforward_nonblocked_ijk_algorithm(int n, double* a, double* b, double* result) {
    gettimeofday(&tv1, &tz);
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += a[n * i + k] * b[n * k + j];
            }
            result[n * i + j] = sum;
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of straightforward non-blocked ijk algorithm - %f\n", elapsed);
}

void blas_blocked_ijk_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result) {
    gettimeofday(&tv1, &tz);
    int i, j, k;
    double alpha = 1.0, beta = 1.0;
    for (i = 0; i < n / cacheBlock; i++) {
        for (j = 0; j < n / cacheBlock; j++) {
            for (k = 0; k < n / cacheBlock; k++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans,
                        CblasNoTrans,
                        cacheBlock, cacheBlock, cacheBlock,
                        alpha, &a[(n * cacheBlock * i) + (cacheBlock * k)],
                        n, &b[(n * cacheBlock * k) + (cacheBlock * j)],
                        n, beta,
                        &result[(n * cacheBlock * i) + (cacheBlock * j)],
                        n);
            }
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of BLAS Blocked ijk algorithm using block size %d - %f\n", cacheBlock, elapsed);
}

void blas_blocked_kij_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result) {
    gettimeofday(&tv1, &tz);
    int k, i, j;
    double alpha = 1.0, beta = 1.0;
    for (k = 0; k < n / cacheBlock; k++) {
        for (i = 0; i < n / cacheBlock; i++) {
            double* block = &a[(n * cacheBlock * i) + (cacheBlock * k)];
            for (j = 0; j < n / cacheBlock; j++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans,
                        CblasNoTrans,
                        cacheBlock, cacheBlock, cacheBlock,
                        alpha, block,
                        n, &b[(n * cacheBlock * k) + (cacheBlock * j)],
                        n, beta,
                        &result[(n * cacheBlock * i) + (cacheBlock * j)],
                        n);
            }
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of BLAS Blocked kij algorithm using block size %d - %f\n", cacheBlock, elapsed);

}

void multiply_nonblocked_kij_algorithm(int n, double* a, double* b,
        double* result) {
    gettimeofday(&tv1, &tz);
    int k, i, j;
    double alpha = 1.0, beta = 1.0;
    for (k = 0; k < n / n; k++) {
        for (i = 0; i < n / n; i++) {
            double* block = &a[(n * n * i) + (n * k)];
            for (j = 0; j < n / n; j++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans,
                        CblasNoTrans,
                        n, n, n,
                        alpha, block,
                        n, &b[(n * n * k) + (n * j)],
                        n, beta,
                        &result[(n * n * i) + (n * j)],
                        n);
            }
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of BLAS nonBlocked kij algorithm  - %f\n", elapsed);

}

void multiply_nonblocked_ijk_algorithm(int n, double* a, double* b,
        double* result) {
    gettimeofday(&tv1, &tz);
    int i, j, k;
    double alpha = 1.0, beta = 1.0;
    for (i = 0; i < n / n; i++) {
        for (j = 0; j < n / n; j++) {
            for (k = 0; k < n / n; k++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans,
                        CblasNoTrans,
                        n, n, n,
                        alpha, &a[(n * n * i) + (n * k)],
                        n, &b[(n * n * k) + (n * j)],
                        n, beta,
                        &result[(n * n * i) + (n * j)],
                        n);
            }
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of BLAS nonBlocked ijk algorithm - %f\n", elapsed);
}

void print_matrix(int n, double* matrix) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.2f", matrix[n * i + j]);
            if (j != n - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
}
