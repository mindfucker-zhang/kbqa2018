#include "dense_layer.h"
#include "tensor_op.h"

int DenseLayer::thread_init(){
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	int size = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		size *= output_shape[i];
	}
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	_bias_flag = true;
	return 0;
}

Tensor DenseLayer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	Tensor& input = inputs["input"];
	DataType* data = _memory_map["output"];
	Tensor ret = TensorOperation::matmul(input, params["weight"], data);
	if (_bias_flag){
		ret += params["bias"];
	}
	return ret;
}

void DenseLayer::set_bias_flag(bool bias_flag){
	_bias_flag = bias_flag;
}

