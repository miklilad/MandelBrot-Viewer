

all: run

compile:
	nvcc -g main.cu -lSDL2

run: compile
	./a.out
