CC = gcc
CFLAGS = -O3 -march=native

BENCHMARKS = compute ptrchase stream

all: $(BENCHMARKS)

compute: compute.c
	$(CC) $(CFLAGS) -o compute compute.c

ptrchase: ptrchase.c
	$(CC) $(CFLAGS) -o ptrchase ptrchase.c

stream: stream.c
	$(CC) $(CFLAGS) -o stream stream.c

clean:
	rm -f $(BENCHMARKS)
