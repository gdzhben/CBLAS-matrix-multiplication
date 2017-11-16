/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   straightforward_nonblocked_ijk.c
 * Author: Ben
 *
 * Created on 25 October 2017, 17:17
 */

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>


struct timeval tv1, tv2;
struct timezone tz;
void blas_blocked_ijk_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result);

/*
 * 
 */
int main(int argc, char** argv) {
    int n;
    int i, j;
    double *a, *b, *c;
    for (n = 2; n < 1000; n *= 2) {
        double* result = (double*) calloc(n*n, sizeof (double));
        a = malloc(sizeof (double)*n * n);
        b = malloc(sizeof (double)*n * n);
        c = malloc(sizeof (double)*n * n);
        srand(time(NULL));
        for (i = 0; i < n * n; i++) {
            a[i] = rand();
            b[i] = rand();
            c[i] = 0;
        }

        for (j = 2; j <= n; j *= j ) {
            blas_blocked_ijk_algorithm(n, j, a, b, result);

        }

        free(a);
        free(b);
        free(c);
    }
    return (EXIT_SUCCESS);
}

void blas_blocked_ijk_algorithm(int n, int cacheBlock, double* a, double* b,
        double* result) {
    gettimeofday(&tv1, &tz);
    int i, j, k;
    double first = 1.0, second = 1.0;
    for (i = 0; i < n / cacheBlock; i++) {
        for (j = 0; j < n / cacheBlock; j++) {
            for (k = 0; k < n / cacheBlock; k++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans,
                        CblasNoTrans,
                        cacheBlock, cacheBlock, cacheBlock,
                        first, &a[(n * cacheBlock * i) + (cacheBlock * k)],
                        n, &b[(n * cacheBlock * k) + (cacheBlock * j)],
                        n, second,
                        &result[(n * cacheBlock * i) + (cacheBlock * j)],
                        n);
            }
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of BLAS Blocked %d * %d matrix ijk algorithm using block size %d - %f\n", n, n, cacheBlock, elapsed);
}

