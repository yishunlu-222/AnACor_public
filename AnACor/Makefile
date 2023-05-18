#CFLAGS = -fPIC -std=c99
#
#all: ray_tracing.so
#
#ray_tracing.so: ray_tracing.c
#	$(CC) -shared -o ray_tracing.so ./c/ray_tracing.c $(CFLAGS)
#
#clean:
#	rm -f ray_tracing.so
CC=gcc
CFLAGS=-Wall -g
pth=./c/
all: program

program:
	$(CC) -shared -o ray_tracing.so ${pth}*.c -fPIC -std=c99


clean:
	rm -f program *.o