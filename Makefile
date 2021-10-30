CXX = dpcpp
CXXFLAGS = -std=c++17 -Wall
SYCLFLAGS = -fsycl
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard include/*.hpp)
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
INCLUDES = -I./include
PROG = run

$(PROG): $(OBJECTS)
	$(CXX) $(SYCLFLAGS) $^ -o $@

ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff.o: ff.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff_p.o: bench_ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff.o: bench_ff.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

clean:
	-rm $(PROG) $(wildcard *.o) $(wildcard include/*.gch)

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
