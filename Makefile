CXX = dpcpp
CXXFLAGS = -std=c++17 -Wall
SYCLFLAGS = -fsycl
SYCLSGFLAGS = -fsycl-default-sub-group-size 32
SYCLCUDAFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
INCLUDES = -I./include
PROG = run

# make file invoker may set shell variable DEVICE to one of possible values {cpu,gpu,host}
# but if nothing is set, `default` is used
#
# anything which is chosen {cpu,gpu,host,default}, is transformed into upper case string
# using shell utility `tr`
#
# @note for more info check `man tr`
#
# @note you must ensure at runtime, binary must have access to cpu/gpu ( as specified during compilation )
# otherwise SYCL runtime will panic, when it won't find desired target device
DFLAGS = -D$(shell echo $(or $(DEVICE),default) | tr a-z A-Z)

$(PROG): main.o utils.o bench_rescue_prime.o bench_ntt.o bench_merkle_tree.o merkle_tree.o ff_p.o rescue_prime.o ntt.o test_ntt.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

test_ntt.o: tests/test_ntt.cpp include/test_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

merkle_tree.o: merkle_tree.cpp include/merkle_tree.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

ntt.o: ntt.cpp include/ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

rescue_prime.o: rescue_prime.cpp include/rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

ff_p.o: ff_p.cpp include/ff_p.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_merkle_tree.o: bench_merkle_tree.cpp include/bench_merkle_tree.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_ntt.o: bench_ntt.cpp include/bench_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_rescue_prime.o: bench_rescue_prime.cpp include/bench_rescue_prime.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

main.o: main.cpp include/bench_rescue_prime.hpp include/bench_ntt.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' -o -name 'lib*.so' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla

tests/ff_p.o: ff_p.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/merkle_tree.o: merkle_tree.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test_merkle_tree.o: tests/test_merkle_tree.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/rescue_prime.o: rescue_prime.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test_rescue_prime.o: tests/test_rescue_prime.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/ntt.o: ntt.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test_ntt.o: tests/test_ntt.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/test.o:	tests/test.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

tests/main.o: tests/main.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ -o $@ $(INCLUDES)

test: tests/ff_p.o tests/merkle_tree.o tests/test_merkle_tree.o tests/rescue_prime.o tests/test_rescue_prime.o tests/ntt.o tests/test_ntt.o tests/test.o tests/main.o
	$(CXX) $(SYCLFLAGS) $^ -o tests/$@
	@./tests/$@

genlib: wrapper/ff_p.o wrapper/ff_p_wrapper.o
	# linking host and pre-compiled device code;
	# producing shared library which will be
	# interacted from example python module `ff_p`
	$(CXX) $(SYCLFLAGS) --shared -fPIC -fsycl-targets=spir64_x86_64 wrapper/*.o -o wrapper/libff_p.so

wrapper/ff_p.o: ff_p.cpp
	# pre-compile kernels targeting CPU
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -fsycl-targets=spir64_x86_64 -c $^ -fPIC -o $@ $(INCLUDES)

wrapper/ff_p_wrapper.o: wrapper/ff_p.cpp
	# pre-compile kernels targeting CPU
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -fsycl-targets=spir64_x86_64 -c $^ -fPIC -o $@ $(INCLUDES)

aot_cpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLSGFLAGS) $(INCLUDES) $(DFLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx512" ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp bench_merkle_tree.cpp merkle_tree.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLSGFLAGS) $(INCLUDES) $(DFLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx2" ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp bench_merkle_tree.cpp merkle_tree.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLSGFLAGS) $(INCLUDES) $(DFLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx" ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp bench_merkle_tree.cpp merkle_tree.cpp utils.o main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLSGFLAGS) $(INCLUDES) $(DFLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=sse4.2" ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp bench_merkle_tree.cpp merkle_tree.cpp utils.o main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLSGFLAGS) $(INCLUDES) $(DFLAGS) -fsycl-targets=spir64_gen -Xs "-device 0x4905" ff_p.cpp bench_rescue_prime.cpp rescue_prime.cpp tests/test_ntt.cpp bench_ntt.cpp ntt.cpp bench_merkle_tree.cpp merkle_tree.cpp utils.o main.o

cuda:
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c main.cpp -o main.o $(DFLAGS)
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c utils.cpp -o utils.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c bench_merkle_tree.cpp -o bench_merkle_tree.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c bench_rescue_prime.cpp -o bench_rescue_prime.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c bench_ntt.cpp -o bench_ntt.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c merkle_tree.cpp -o merkle_tree.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c ff_p.cpp -o ff_p.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c ntt.cpp -o ntt.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c rescue_prime.cpp -o rescue_prime.o
	clang++ $(CXXFLAGS) $(SYCLCUDAFLAGS) $(INCLUDES)  -c tests/test_ntt.cpp -o test_ntt.o
	clang++ $(SYCLCUDAFLAGS) *.o -o $(PROG)
