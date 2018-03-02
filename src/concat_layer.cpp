#include <iostream>
#include "concat_layer.h"
#include "tensor_op.h"

int ConcatLayer::thread_init(){
	_axis = _param_conf["axis"][0];
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	int size = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		size *= output_shape[i];
	}
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	return 0;
}

Tensor ConcatLayer::process(std::map<std::string, Tensor>& inputs){
	Tensor t1 = inputs["input1"];
	Tensor t2 = inputs["input2"];
	TensorOperation::concat(t1, t2, _axis, _memory_map["output"]);
	return _tensor_map["output"];
}
