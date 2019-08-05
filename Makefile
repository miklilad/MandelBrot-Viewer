

all: run

compile:
	g++ -Wall -pedantic main.cpp -lSDL2

run: compile
	./a.out
