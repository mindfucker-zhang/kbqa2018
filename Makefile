qa_test:
	g++ src/*.cpp -I ./include/ -Wall -Wunused -DOS_LINUX -std=c++11 -lNLPIR -lboost_serialization  -lpthread -g -O3  -msse3 -o run_env/bin/qa_test test/qa_test.cpp
