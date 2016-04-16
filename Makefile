CC = g++

GFLAG = -O2

EXEC = handwritten_recognition

SRC = $(wildcard *.cpp)
OBJ = network.o handwritten_recognition.o

all:  $(OBJ)
	$(CC) $(OBJ) $(GFLAG) -o $(EXEC)

run: 
	./$(EXEC)

%.o: %.cpp
	$(CC) -c $<  $(GFLAG)

clean:
	rm -f *.o