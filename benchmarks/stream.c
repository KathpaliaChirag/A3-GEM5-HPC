#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main(int argc, char** argv){
    size_t n = (argc>1)? strtoull(argv[1],NULL,10) : 2000000;

    double *A = malloc(n * sizeof(double));
    double *B = malloc(n * sizeof(double));
    double *C = malloc(n * sizeof(double));
    if(!A || !B || !C){
        fprintf(stderr,"malloc fail\n");
        return 1;
    }

    for(size_t i=0;i<n;i++){
        A[i]=1.0;
        B[i]=2.0;
        C[i]=0.0;
    }

    // STREAM-like triad: C = A + alpha*B
    double alpha = 3.14159;
    for(size_t t=0;t<100; t++){   // repeat to increase runtime
        for(size_t i=0;i<n;i++){
            C[i] = A[i] + alpha*B[i];
        }
    }

    printf("done %f\n", C[n-1]);
    free(A); free(B); free(C);
    return 0;
}
