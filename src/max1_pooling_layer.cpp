#include "max1_pooling_layer.h"
#include "tensor_op.h"
#include "logger.h"

int Max1PoolingLayer::thread_init(){
	const std::vector<int>& input_shape = _param_conf["input_shape"];
	_axis = _param_conf["axis"][0];
	std::vector<int> output_shape;
	int size = 1;
	for (int i = 0; i < _axis; ++i){
		output_shape.push_back(input_shape[i]);
		size *= input_shape[i];
	}
	for (int i = _axis + 1; i < input_shape.size(); ++i){
		output_shape.push_back(input_shape[i]);
		size *= input_shape[i];
	}
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	return 0;
}

Tensor Max1PoolingLayer::process(std::map<std::string, Tensor>& inputs){
	Tensor& input = inputs["input"];
	Tensor& output = _tensor_map["output"];
	TensorOperation::max(input, _axis, output.get_var_data());
	return output;
}
