CC = g++
DIALECT = c++17 # be explicit
OPENCV = `pkg-config opencv4 --cflags --libs`
GCC_OPTIONS = -Wall -std=$(DIALECT)
CFLAGS = $(GCC_OPTIONS) -fopenmp $(FILE)$(EXT) -o $(FILE) $(OPENCV)
DEBUG_FLAG = -g
FILE = sobel
TEST_IMG = ../test_data/huge.jpg
DEBUGGER = gdb
DEBUGGER_ARGS = --args $(FILE) $(TEST_IMG)
EXT = .cpp

build: clean
	$(CC) $(CFLAGS)

start:
	./$(FILE) $(TEST_IMG)

test: build start

debugbuild: clean
	$(CC) $(DEBUG_FLAG) $(CFLAGS)

debug: debugbuild
	$(DEBUGGER) $(DEBUGGER_ARGS)

clean:
	rm -f $(FILE)