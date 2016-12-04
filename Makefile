CC = g++
PYTHON = python3.5

OMP_NUM_THREADS?=3
GFLAG = -O3 -Wall -Wextra -fopenmp -DOMP_NUM_THREADS=$(OMP_NUM_THREADS)

EXEC = handwritten_recognition

SRCNET= $(wildcard Network/*.cpp)
SRCEXT= $(wildcard Extraction/*.cpp)
OBJ= $(SRCNET:%.cpp=%.o) $(SRCEXT:%.cpp=%.o) $(EXEC).o

download:
	$(PYTHON) Data/download.py
	gunzip Data/*.gz

all:  $(OBJ)
	$(CC) $(OBJ) $(GFLAG) -o $(EXEC)

run: all
	./$(EXEC)

%.o: %.cpp
	$(CC) -c $<  $(GFLAG) -o $@

clean:
	rm -f $(OBJ) $(EXEC)

full: clean all run
