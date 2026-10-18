OPENBLAS_HOME ?= $(HOME)/.local/openblas
CXX ?= g++
CC ?= gcc
CPPFLAGS = -I$(OPENBLAS_HOME)/include
CXXFLAGS = -O2 -std=c++17
CFLAGS = -O2 -fPIC
LDFLAGS = -L$(OPENBLAS_HOME)/lib -Wl,-rpath,$(OPENBLAS_HOME)/lib -lopenblas -ldl

.PHONY: all clean
all: gesvd_replay libsvd_count.so

gesvd_replay: gesvd_replay.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

libsvd_count.so: svd_count.c
	$(CC) $(CFLAGS) -shared -o $@ $< -ldl

clean:
	rm -f gesvd_replay libsvd_count.so
