#include <sstream>
#include "data_feed.h"
#include "logger.h"
#include "utils.h"

int DataFeed::thread_init(const std::map<std::string, std::string>& conf){
	auto iter = conf.find("batch_size");
	if (iter == conf.end()){
		Logger::logging("data feed batch size error!", "ERROR");
		return -1;
	}
	_batch_size = Utils::str2int(iter->second);
	if (!_batch_size){
		Logger::logging("data feed batch size error!", "ERROR");
		return -1;
	}
	_batch_input.resize(_batch_size);	
	return 0; 
}

void DataFeed::thread_destroy(){
	for (int i = 0; i < _memory_pool.size(); ++i){
		if (_memory_pool[i]){
			delete[]  _memory_pool[i];
		}
	}
	_memory_pool.clear();
	_batch_input.clear();
	_batch_size = 0;
}


std::vector<std::map<std::string, Tensor> >& DataFeed::gen_batch_input(
		std::vector<void*>& batch_org_data, void* dict){
	int batch_size = _batch_size;
	if (batch_org_data.size() < batch_size){
		batch_size = batch_org_data.size();
	}
	for (int i = 0; i < batch_size; ++i){
		gen_input(batch_org_data[i], dict, i);
	}
	return _batch_input;
}	
