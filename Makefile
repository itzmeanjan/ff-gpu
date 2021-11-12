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

rescue_prime.o: rescue_prime.cpp include/rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff_p.o: ff_p.cpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff.o: ff.cpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_rescue_prime.o: bench_rescue_prime.cpp include/bench_rescue_prime.hpp include/rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff_p.o: bench_ff_p.cpp include/bench_ff_p.hpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff.o: bench_ff.cpp include/bench_ff.hpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

main.o: main.cpp include/bench_ff.hpp include/bench_ff_p.hpp include/bench_rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

tests/ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/rescue_prime.o: rescue_prime.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test_rescue_prime.o: tests/test_rescue_prime.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/ntt.o: ntt.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test_ntt.o: tests/test_ntt.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test.o:	tests/test.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -o $@ $(INCLUDES)

test: tests/ff_p.o tests/rescue_prime.o tests/test_rescue_prime.o tests/ntt.o tests/test_ntt.o tests/test.o tests/main.o
	$(CXX) $(SYCLFLAGS) $^ -o tests/$@
	@./tests/$@

genlib: wrapper/ff_p.o wrapper/ff_p_wrapper.o
	$(CXX) $(SYCLFLAGS) --shared -fPIC wrapper/*.o -o wrapper/libff_p.so

wrapper/ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -fPIC -o $@ $(INCLUDES)

wrapper/ff_p_wrapper.o: wrapper/ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ -fPIC -o $@ $(INCLUDES)

aot_cpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp utils.o main.o

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device 0x4905" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp utils.o main.o
