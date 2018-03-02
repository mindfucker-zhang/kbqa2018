qa_test:
	g++ src/*.cpp -I ./include/ -Wall -Wunused -DOS_LINUX -std=c++11 -lNLPIR -lboost_serialization  -lpthread -g -O3  -msse4.2 -o run_env/bin/qa_test test/qa_test.cpp

mention_test:
	g++ src/*.cpp -I ./include/ -Wall -Wunused -DOS_LINUX -std=c++11 -lNLPIR -lboost_serialization  -lpthread -g -O3  -msse4.2 -o run_env/bin/mention_test test/mention_test.cpp

cnn_test:
	g++ src/*.cpp -I ./include/ -Wall -Wunused -DOS_LINUX -std=c++11 -lNLPIR -lboost_serialization  -lpthread -g -O3  -msse4.2 -o run_env/cnn_test test/cnn_rc.cpp
