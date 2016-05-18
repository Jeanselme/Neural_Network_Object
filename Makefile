CC = g++

# Delete -DSTOCHASTIC  if a real gradient descent is needed
GFLAG = -O3 -Wall -Wextra -DSTOCHASTIC 

EXEC = handwritten_recognition

SRC = $(wildcard *.cpp)
OBJ = activation.o network.o extraction.o $(EXEC).o

all:  $(OBJ)
	$(CC) $(OBJ) $(GFLAG) -o $(EXEC)

run: 
	./$(EXEC)

%.o: %.cpp
	$(CC) -c $<  $(GFLAG)

clean:
	rm -f *.o *.gch

full: clean all run