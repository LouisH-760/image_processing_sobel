CC = nvcc
OPENCV = `pkg-config opencv4 --cflags --libs`
GCC_OPTIONS = --compiler-options -Wall
CFLAGS = $(OPENCV) $(GCC_OPTIONS) $(FILE)$(EXT) -o $(FILE)
DEBUG_FLAG = -g
FILE = sobel
TEST_IMG = ../test_data/huge.jpg
DEBUGGER = gdb
DEBUGGER_ARGS = --args $(FILE) $(TEST_IMG)
EXT = .cu

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