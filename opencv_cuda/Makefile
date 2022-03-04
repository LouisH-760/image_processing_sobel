CC = nvcc
DIALECT = gnu18 # be explicit
OPENCV = `pkg-config opencv4 --cflags --libs`
GCC_OPTIONS = --compiler-options -Wall --compiler-options -std=$(DIALECT)
CFLAGS = $(OPENCV) $(GCC_OPTIONS)
FILE = sobel
TEST_IMG = ../test_data/huge.jpg

build: clean
	$(CC) $(CFLAGS) $(FILE).cu -o $(FILE)

start:
	./$(FILE) $(TEST_IMG)

test: build start

clean:
	rm -f sobel