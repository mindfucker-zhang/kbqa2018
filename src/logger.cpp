#include <iostream>
#include "logger.h"

pthread_mutex_t Logger::_s_logger_mutex;

int Logger::global_init(){
	return pthread_mutex_init(&_s_logger_mutex, NULL);
}

void Logger::logging(const std::string& info, const std::string& type){
	pthread_mutex_lock(&_s_logger_mutex);
	std::cerr << "[" << type << "]: "  << info << std::endl;
	pthread_mutex_unlock(&_s_logger_mutex);
}




