CC = g++
PYTHON = python3.5

# Display = 1 -> True 0 -> False
VERBOSE?=1
TIME?=1
# Size of batch
BATCH?=100
# Number of iterations
MAXITER?=50
#Number of threads
OMP_NUM_THREADS?=3
GFLAG = -O3 -Wall -Wextra -fopenmp -DOMP_NUM_THREADS=$(OMP_NUM_THREADS) -DVERBOSE=$(VERBOSE) -DTIME=$(TIME) -DSIZE_BATCH=$(BATCH) -DMAX_ITERATION=$(MAXITER)

EXEC = handwritten_recognition

SRCNET= $(wildcard Network/*.cpp)
SRCEXT= $(wildcard Extraction/*.cpp)
OBJ= $(SRCNET:%.cpp=%.o) $(SRCEXT:%.cpp=%.o) $(EXEC).o

test:
	$(PYTHON) Performances.py

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
