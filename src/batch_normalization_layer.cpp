#include <iostream>
#include "batch_normalization_layer.h"
#include "tensor_op.h"

int BatchNormalizationLayer::thread_init(){
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	int size = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		size *= output_shape[i];
	}
	int feat_num = output_shape[output_shape.size() - 1];
	
	std::vector<int> std_dev_shape;
	std_dev_shape.push_back(feat_num);

	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	_memory_map["std_dev"] = new DataType[feat_num];
	_tensor_map["std_dev"] = Tensor(std_dev_shape, false, _memory_map["std_dev"]);
	return 0;
}

Tensor BatchNormalizationLayer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	Tensor& input = inputs["input"];
	Tensor& std_dev = _tensor_map["std_dev"];
	Tensor& output = _tensor_map["output"];
	Tensor& var = params["var"];
	Tensor& mean = params["mean"];
	Tensor& scale = params["scale"];
	Tensor& offset = params["offset"];
	Tensor& epsilon = params["epsilon"];
	
	TensorOperation::add(var, epsilon, std_dev.get_var_data());
	
	TensorOperation::sqrt_t(std_dev, std_dev.get_var_data());
	
	DataType* output_data = output.get_var_data();
	TensorOperation::sub(input, mean, output_data);

	output /= std_dev;
	output *= scale;
	output += offset;
	return output;
}
