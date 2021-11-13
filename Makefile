CXX = dpcpp
CXXFLAGS = -std=c++17 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run

$(PROG): main.o utils.o bench_ff.o bench_ff_p.o bench_rescue_prime.o bench_ntt.o ff.o ff_p.o rescue_prime.o ntt.o test_ntt.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

test_ntt.o: tests/test_ntt.cpp include/test_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ntt.o: ntt.cpp include/ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

rescue_prime.o: rescue_prime.cpp include/rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff_p.o: ff_p.cpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

ff.o: ff.cpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ntt.o: bench_ntt.cpp include/bench_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_rescue_prime.o: bench_rescue_prime.cpp include/bench_rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff_p.o: bench_ff_p.cpp include/bench_ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

bench_ff.o: bench_ff.cpp include/bench_ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

main.o: main.cpp include/bench_ff.hpp include/bench_ff_p.hpp include/bench_rescue_prime.hpp include/bench_ntt.hpp
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
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx512" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp utils.o main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=sse4.2" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp utils.o main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device 0x4905" ff.cpp bench_ff.cpp ff_p.cpp bench_ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp utils.o main.o
