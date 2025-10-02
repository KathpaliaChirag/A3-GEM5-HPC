#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef struct Node {
    uint32_t next;
    uint32_t pad[7]; // padding to make nodes larger (simulate cache lines)
} Node;

int main(int argc, char** argv){
    size_t n = (argc>1)? strtoull(argv[1],NULL,10) : 200000;
    Node *A = malloc(n * sizeof(Node));
    if(!A){
        fprintf(stderr,"malloc fail\n");
        return 1;
    }

    // initialize as a simple linked list
    for(size_t i=0;i<n;i++)
        A[i].next = (i+1)%n;

    // randomize the pointers (shuffle the list)
    srand((unsigned)time(NULL));
    for(size_t i=n-1;i>0;i--){
        size_t j = rand() % (i+1);
        uint32_t t = A[i].next;
        A[i].next = A[j].next;
        A[j].next = t;
    }

    // pointer chasing loop
    uint32_t cur=0;
    for(size_t t=0; t<1000000ULL; t++)
        cur = A[cur].next;

    printf("end %u\n", cur);
    free(A);
    return 0;
}
