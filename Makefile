CC = gcc
CFLAGS = -g -Wall -lm 


NN = nn
NN_LIB = nn_library
EX_CR = create_examples
LOAD_EX = load_examples
MATRIX = Matrix

default: all


all : $(NN)

$(NN): $(NN).o $(EX_CR).o $(MATRIX).o $(NN_LIB).o $(LOAD_EX).o
	$(CC)  -o $(NN) *.o  $(CFLAGS)

$(NN).o: $(NN).c
	$(CC)  -c $(NN).c $(CFLAGS)

$(EX_CR).o: $(EX_CR).c $(EX_CR).h 
	$(CC)  -c $(EX_CR).c $(CFLAGS)

$(LOAD_EX).o: $(LOAD_EX).c $(LOAD_EX).h 
	$(CC)  -c $(LOAD_EX).c $(CFLAGS)

$(NN_LIB).o: $(NN_LIB).c $(NN_LIB).h 
	$(CC)  -c $(NN_LIB).c $(CFLAGS)

$(MATRIX).o: $(MATRIX).c $(MATRIX).h 
	$(CC)  -c $(MATRIX).c $(CFLAGS)


cleanAll: cleanFiles clean

clean: 
	$(RM) $(NN) 
	$(RM) *.o


cleanFiles:
	$(RM) *.txt

hello: 
	echo "hello world"