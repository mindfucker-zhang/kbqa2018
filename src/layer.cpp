#include "layer.h"

Layer::~Layer(){
	thread_destroy();
}

Tensor Layer::process(std::map<std::string, Tensor>& inputs){
	return Tensor();
}

Tensor Layer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	return Tensor();
}

void Layer::set_param_conf(const std::map<std::string, std::vector<int> >& param_conf){
	_param_conf = param_conf;
}

void Layer::thread_destroy(){
	std::map<std::string, DataType*>::iterator iter = _memory_map.begin(); 	
	for (; iter != _memory_map.end(); ++iter){
		if (iter->second){
			delete[] (iter->second);
			iter->second = NULL;
		}
	}
}

Tensor Layer::get_tensor_by_key(const std::string& key){
	auto iter = _tensor_map.find(key);
	if (iter == _tensor_map.end()){
		return Tensor();
	}
	return iter->second;
}

int Layer::get_tensor_shape(const std::string& key, std::vector<int>& ret_shape){
	auto iter = _tensor_map.find(key);
	if (iter == _tensor_map.end()){
		return -1;
	}
	ret_shape = iter->second.get_shape();
	return 0;	
}
