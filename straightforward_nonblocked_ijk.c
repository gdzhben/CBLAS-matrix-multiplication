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

void straightforward_nonblocked_ijk_algorithm(int n, double* a, double* b, double *r);

/*
 * 
 */
int main(int argc, char** argv) {
    int n;
    int i;
    double *a, *b, *c;
    for (n = 2; n < 1025; n *= 2) {
        a = malloc(sizeof (double)*n * n);
        b = malloc(sizeof (double)*n * n);
        c = malloc(sizeof (double)*n * n);
        srand(time(NULL));
        for (i = 0; i < n * n; i++) {
            a[i] = rand();
            b[i] = rand();
            c[i] = 0;
        }
        straightforward_nonblocked_ijk_algorithm(n, a, b, c);
        free(a);
        free(b);
        free(c);
    }
    return (EXIT_SUCCESS);
}

void straightforward_nonblocked_ijk_algorithm(int n, double* a, double* b, double *r) {
    gettimeofday(&tv1, &tz);
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += a[n * i + k] * b[n * k + j];
            }
            r[n * i + j] = sum;
        }
    }
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec - tv1.tv_sec) + (double) (tv2.tv_usec - tv1.tv_usec) * 1.e-6;
    printf("Benchmark of straightforward non-blocked %d * %d matrix ijk algorithm - %f\n", n, n, elapsed);
}