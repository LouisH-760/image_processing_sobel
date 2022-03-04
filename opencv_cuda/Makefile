CC = nvcc
OPENCV = `pkg-config opencv4 --cflags --libs`
CFLAGS = $(OPENCV) --compiler-options -Wall
FILE = sobel
TEST_IMG = ../test_data/huge.jpg

build: clean
	$(CC) $(CFLAGS) $(FILE).cu -o $(FILE)

start:
	./$(FILE) $(TEST_IMG)

test: build start

clean:
	rm sobel