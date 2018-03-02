#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <pthread.h>
#include <string>

class Logger{

public:
	static int global_init();
	static void logging(const std::string& info, const std::string& type);
	static pthread_mutex_t _s_logger_mutex;
};

#endif
