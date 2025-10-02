#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv){
    const uint64_t N = (argc>1)? strtoull(argv[1],NULL,10) : 100000ULL;
    double x=1.234567;
    for(uint64_t i=0;i<N;i++){
        x = x*1.000001 + 0.0000001;
        x = x/1.000001 + 0.0000002;
        x += 3.14159;
    }
    printf("done %f\n", x);
    return 0;
}
