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
GPROF = gprof
GPROF_ARGS = $(FILE) gmon.out
GPROF_FLAG = -pg
PROFILER = valgrind
PROFILER_ARGS = --callgrind-out-file=$(CALLGRIND_FILE) --tool=callgrind ./$(FILE) $(TEST_IMG)
CALLGRIND_FILE = callgrind.out



build: clean
	$(CC) $(CFLAGS)

start:
	./$(FILE) $(TEST_IMG)

test: build start

debugbuild: clean
	$(CC) $(DEBUG_FLAG) $(CFLAGS)

debug: debugbuild
	$(DEBUGGER) $(DEBUGGER_ARGS)

gprofbuild: clean
	$(CC) $(GPROF_FLAG) $(CFLAGS)

gprof: gprofbuild start
	$(GPROF) $(GPROF_ARGS)

profcall: debugbuild
	$(PROFILER) $(PROFILER_ARGS)

profile: profcall
	kcachegrind $(CALLGRIND_FILE)

clean:
	rm -f $(FILE)
	rm -f $(CALLGRIND_FILE)
	rm -f gmon.out
	rm -f result.png