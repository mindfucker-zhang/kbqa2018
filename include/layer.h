#ifndef _LAYER_H_
#define _LAYER_H_
#include <map>
#include <vector>
#include <string>
#include "tensor.h"

class Layer{

public:
	virtual ~Layer();
	void set_param_conf(const std::map<std::string, std::vector<int> >& param_conf);
	virtual int thread_init() = 0;
	void thread_destroy();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);
	virtual Tensor process(std::map<std::string, Tensor>& inputs);
	Tensor get_tensor_by_key(const std::string& key);
	int get_tensor_shape(const std::string& key, std::vector<int>& ret_shape);
	
protected:
	std::map<std::string, std::vector<int> > _param_conf;
	std::map<std::string, DataType*> _memory_map;
	std::map<std::string, Tensor> _tensor_map; 
};

#endif
