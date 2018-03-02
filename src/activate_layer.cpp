#include "activate_layer.h"
#include "tensor_op.h"

void ActivateLayer::set_func_key(const std::string& func_key){
	_func_key = func_key;
}

int ActivateLayer::thread_init(){
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	if (!output_shape.size() || !output_shape[0]){
		_in_situ_tag = true;
		return 0;
	}

	_in_situ_tag = false;
	int size = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		size *= output_shape[i];
	}
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	return 0;
}

Tensor ActivateLayer::process(std::map<std::string, Tensor>& inputs){
	Tensor& input = inputs["input"];
	if (_in_situ_tag){
		TensorOperation::activation_func_in_situ(input, _func_key);
		return input;
	}
	Tensor& output = _tensor_map["output"];
	DataType* output_data = output.get_var_data();
	TensorOperation::activation_func(input, _func_key, output_data);
	return output;
}
