.PHONY: all
all: mnist.exe
all: mnist_dag.exe
all: mnist_gui.exe

CC = gcc
CARGS = -I../../ -std=c23 -Wall -Wextra -D_CRT_SECURE_NO_WARNINGS# -D ML2_PRINT_ALLOC
OPTIM = -Ofast -march=native -D ML2_NO_DEBUG
DEBUG = -g
OMP = -fopenmp

mnist.exe: mnist.c
	$(CC) $^ -o $@ $(CARGS) $(OPTIM) $(OMP) -fwhole-program

mnist_dag.exe: mnist_dag.c
	$(CC) $^ -o $@ $(CARGS) $(OPTIM) $(OMP) -fwhole-program

mnist_gui.exe: mnist_gui.c
	$(CC) $^ -o $@ $(CARGS) $(OPTIM) $(OMP) -mwindows
