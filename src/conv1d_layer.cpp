#include <cstring>
#include <iostream>
#include "conv1d_layer.h"
#include "tensor_op.h"
#include "logger.h"

int Conv1dLayer::thread_init(){
	const std::vector<int>& input_shape = _param_conf["input_shape"];
	if (input_shape.size() < 2){
		Logger::logging("Conv1d input_shape para error!", "ERROR");
		return -1;
	}
	
	int dim = input_shape.size();
	_in_seq_len = input_shape[dim - 2];
	_in_feat_num = input_shape[dim - 1];
	_win_size = _param_conf["win_size"][0];
	_out_feat_num = _param_conf["out_feat_size"][0];
	_iters = 1;
	std::vector<int> output_shape;
	std::vector<int> expanding_input_shape;
	output_shape.resize(dim);
	expanding_input_shape.resize(dim);
	for (int i = 0; i < input_shape.size() - 2; ++i){
		_iters *= input_shape[i];
		output_shape[i] = input_shape[i];
		expanding_input_shape[i] = input_shape[i];
	}
	output_shape[dim -2] = _in_seq_len;
	output_shape[dim -1] = _out_feat_num;
	expanding_input_shape[dim - 2] = _in_seq_len;
	expanding_input_shape[dim - 1] = _win_size * _in_feat_num;

	_memory_map["output"] = new DataType[_iters * _in_seq_len * _out_feat_num];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	_memory_map["expanding_input"] = new DataType[_iters * _in_seq_len * _win_size * _in_feat_num];
	_tensor_map["expanding_input"] = Tensor(expanding_input_shape,
			false, _memory_map["expanding_input"]);
	_bias_flag = true;
	return 0;
}

Tensor Conv1dLayer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	Tensor& input = inputs["input"];
	Tensor& expanding_input = _tensor_map["expanding_input"];
	Tensor& output = _tensor_map["output"];

	DataType* input_data = input.get_var_data();
	DataType* expanding_data = expanding_input.get_var_data();
	DataType* output_data = output.get_var_data();

	int iter_size = _in_seq_len * _in_feat_num;
	int left = (_win_size - 1) / 2;
	int eindex = 0;
	for (int i = 0; i < _iters; ++i){
		for (int j = -left; j < 0; ++ j){
			int zero_size = -j * _in_feat_num;
			memset(expanding_data + eindex, 0, zero_size * sizeof(DataType));
			eindex += zero_size;
			int real_win_size = _win_size + j;
			memcpy(expanding_data + eindex, input_data, real_win_size *
					_in_feat_num * sizeof(DataType));
			eindex += real_win_size * _in_feat_num;
		}


		for (int j = 0; j < _in_seq_len - left; ++j){
			int input_st_index = i * iter_size + j * _in_feat_num;
			int real_win_size = _in_seq_len - j;
			real_win_size = (_win_size < real_win_size)? _win_size: real_win_size;
			memcpy(expanding_data + eindex, input_data + input_st_index, 
					real_win_size * _in_feat_num * sizeof(DataType));
			eindex += real_win_size * _in_feat_num;
			int zero_size = (_win_size - real_win_size) * _in_feat_num;
			if (zero_size){
				memset(expanding_data + eindex, 0, zero_size * sizeof(DataType));
				eindex += zero_size;
			}
		}
	}
	
	TensorOperation::matmul(expanding_input, params["weight"], output_data); 
	if (_bias_flag){
		output += params["bias"];
	}
	return output;
}

void Conv1dLayer::set_bias_flag(bool bias_flag){
	_bias_flag = bias_flag;
}
