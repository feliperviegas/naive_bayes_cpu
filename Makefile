CC = gcc
CCC= g++
GPUCC= nvcc
PARAMETERS = -O3 -lm
CFLAGS = -arch sm_30

main: main.o io.o nb_functions.o
	$(CCC) $(PARAMETERS) -o nb main.o io.o nb_functions.o

main.o: main.cpp
	$(CCC) $(PARAMETERS) -c main.cpp -o main.o

io.o: io.cpp io.hpp
	$(CCC) $(PARAMETERS) -c io.cpp -o io.o

nb_functions.o: nb_functions.cpp nb_functions.hpp
	$(CCC)  -c nb_functions.cpp -o nb_functions.o


clean:
	rm -f *.o nb *~
