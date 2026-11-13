.PHONY: all
all: mnist.exe
all: mnist_gui.exe

CC = gcc
CARGS = -I../../ -std=c23 -Wall -Wextra -Wno-unused-parameter -D_CRT_SECURE_NO_WARNINGS# -DML2_PRINT_ALLOC
OPTIM = -Ofast -march=native# -DNDEBUG -DML2_NO_DEBUG -DML2_NO_DEBUG_AT_ALL
DEBUG = -g
OMP = -fopenmp

mnist.exe: mnist.c
	$(CC) $^ -o $@ $(CARGS) $(OPTIM) $(OMP) -fwhole-program

mnist_gui.exe: mnist_gui.c
	$(CC) $^ -o $@ $(CARGS) $(OPTIM) $(OMP) -mwindows
