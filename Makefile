CXX = dpcpp
CXXFLAGS = -std=c++17 -Wall
SYCLFLAGS = -fsycl
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard include/*.hpp)
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
INCLUDES = -I./include

ff.o: ff.cpp include/ff.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c $^ $(INCLUDES)

clean:
	-rm $(wildcard *.o) $(wildcard include/*.gch)
