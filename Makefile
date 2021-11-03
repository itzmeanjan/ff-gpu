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

ff_p.o: ff_p.cpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff.o: ff.cpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff_p.o: bench_ff_p.cpp include/bench_ff_p.hpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff.o: bench_ff.cpp include/bench_ff.hpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

main.o: main.cpp include/bench_ff.hpp include/bench_ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

clean:
	-rm $(PROG) $(wildcard *.o) $(wildcard tests/*.o) $(wildcard wrapper/*.*o) $(wildcard include/*.gch)

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

tests/ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test.o:	tests/test.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

test: tests/ff_p.o tests/test.o tests/main.o
	$(CXX) $(SYCLFLAGS) $^ -o tests/$@
	@./tests/$@

genlib: wrapper/ff_p.o wrapper/ff_p_wrapper.o
	$(CXX) $(SYCLFLAGS) --shared -fPIC wrapper/*.o -o wrapper/libff_p.so

wrapper/ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -fPIC -o $@ $(INCLUDES)

wrapper/ff_p_wrapper.o: wrapper/ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -fPIC -o $@ $(INCLUDES)
