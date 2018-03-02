#include <math.h>
#include "softmax_layer.h"

int SoftmaxLayer::thread_init(){
	_axis = _param_conf["axis"][0];	
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	_iters = 1;
	_ave_num = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		if (i < _axis){
			_iters *= output_shape[i];
		}else{
			_ave_num *= output_shape[i];
		}
	}
	int size = _iters * _ave_num;
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	return 0;
}

Tensor SoftmaxLayer::process(std::map<std::string, Tensor>& inputs,
                std::map<std::string, Tensor>& params){
	Tensor& input = inputs["input"];
	Tensor& output = _tensor_map["output"];
	const DataType* in_data = input.get_data();
	DataType* out_data = output.get_var_data();
	for (int i = 0; i < _iters; ++i){
		DataType sum = 0;
		for (int j = 0; j < _ave_num; ++j){
			sum += exp(in_data[i * _ave_num + j]);
		}
		for (int j = 0; j < _ave_num; ++j){
			out_data[i * _ave_num + j] = exp(in_data[i * _ave_num + j]) / sum;
		}
	}
	return _tensor_map["output"];
}
