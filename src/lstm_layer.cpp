#include "lstm_layer.h"
#include "tensor_op.h"

void LstmLayer::set_reversed(bool flag){
	_reversed_flag = flag;
}

int LstmLayer::thread_init(){
	const std::vector<int>& input_shape = _param_conf["input_shape"];
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	int seq_len = input_shape[0];
	int input_dim = input_shape[1];
	int hidden_dim = output_shape[1];
	_seq_len = seq_len;

	std::vector<int> product_shape;	
	product_shape.push_back(hidden_dim);
	
	_memory_map["input_product"] = new DataType[hidden_dim];
	_tensor_map["input_product"] = Tensor(product_shape, false, _memory_map["input_product"]);
	_memory_map["output_product"] = new DataType[hidden_dim];
	_tensor_map["output_product"] = Tensor(product_shape, false, _memory_map["output_product"]);
	_memory_map["last_product"] = new DataType[hidden_dim];
	_tensor_map["last_product"] = Tensor(product_shape, false, _memory_map["last_product"]);
        _memory_map["new_product"] = new DataType[hidden_dim];
	_tensor_map["new_product"] = Tensor(product_shape, false, _memory_map["new_product"]);
	
	_memory_map["last_state"] = new DataType[hidden_dim];
	_tensor_map["last_state"] = Tensor(product_shape, false, _memory_map["last_state"]);
	_memory_map["last_output"] = new DataType[hidden_dim];
        _tensor_map["last_output"] = Tensor(product_shape, false, _memory_map["last_output"]);
	
	std::vector<int> gate_shape;
	gate_shape.push_back(seq_len);
	gate_shape.push_back(hidden_dim);
	int gate_size = seq_len * hidden_dim;

	_memory_map["input_gate"] = new DataType[gate_size];
	_tensor_map["input_gate"] = Tensor(gate_shape, false, _memory_map["input_gate"]);
        _memory_map["forget_gate"] = new DataType[gate_size];
	_tensor_map["forget_gate"] = Tensor(gate_shape, false, _memory_map["forget_gate"]);
        _memory_map["output_gate"] = new DataType[gate_size];
	_tensor_map["output_gate"] = Tensor(gate_shape, false, _memory_map["output_gate"]);
        _memory_map["tmp_state"] = new DataType[gate_size];
	_tensor_map["tmp_state"] = Tensor(gate_shape, false, _memory_map["tmp_state"]);

        _memory_map["state"] = new DataType[gate_size];
	_tensor_map["state"] = Tensor(gate_shape, false, _memory_map["state"]);
	_memory_map["output"] = new DataType[gate_size];
	_tensor_map["output"] = Tensor(gate_shape, false, _memory_map["output"]);
	return 0;
}

Tensor LstmLayer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	Tensor& input = inputs["input"];
	Tensor& input_gate_input_weight = params["input_gate_input_weight"];
	Tensor& input_gate_output_weight = params["input_gate_output_weight"];
	Tensor& input_gate_bias = params["input_gate_bias"];
	Tensor& forget_gate_input_weight = params["forget_gate_input_weight"];
        Tensor& forget_gate_output_weight = params["forget_gate_output_weight"];
        Tensor& forget_gate_bias = params["forget_gate_bias"];
	Tensor& forget_gate_bias_const = params["forget_gate_bias_const"];
	Tensor& output_gate_input_weight = params["output_gate_input_weight"];
        Tensor& output_gate_output_weight = params["output_gate_output_weight"];
        Tensor& output_gate_bias = params["output_gate_bias"];
	Tensor& state_input_weight = params["state_input_weight"];
        Tensor& state_output_weight = params["state_output_weight"];
        Tensor& state_bias = params["state_bias"];

	Tensor& input_product = _tensor_map["input_product"];
	Tensor& output_product = _tensor_map["output_product"];
	Tensor& last_product = _tensor_map["last_product"];
	Tensor& new_product = _tensor_map["new_product"];
	Tensor& last_state = _tensor_map["last_state"];
	Tensor& last_output = _tensor_map["last_output"];
	Tensor& input_gate = _tensor_map["input_gate"];
	Tensor& forget_gate = _tensor_map["forget_gate"];
	Tensor& output_gate = _tensor_map["output_gate"];
	Tensor& tmp_state = _tensor_map["tmp_state"];
	Tensor& state = _tensor_map["state"];
	Tensor& output = _tensor_map["output"];
	
	int st = 0;
	int en = _seq_len;
	int step = 1;
	if (_reversed_flag){
		st = _seq_len -1;
		en = -1;
		step = -1;
	}
	std::vector<int> index;
	index.resize(1);
	last_state.zero();
	last_output.zero();
	for (int i = st; i != en; i += step){
		index[0] = i;
		Tensor cur_input = input.indexing(index); 
		Tensor cur_input_gate = input_gate.indexing(index);
		Tensor cur_forget_gate = forget_gate.indexing(index);
		Tensor cur_output_gate = output_gate.indexing(index); 
		Tensor cur_tmp_state = tmp_state.indexing(index);
		Tensor cur_state = state.indexing(index);
		Tensor cur_output = output.indexing(index);

		TensorOperation::matmul(cur_input, input_gate_input_weight, 
				input_product.get_var_data());
		TensorOperation::matmul(last_output, input_gate_output_weight,
				output_product.get_var_data());
		TensorOperation::add(input_product, output_product,
				cur_input_gate.get_var_data());
		cur_input_gate += input_gate_bias;
		TensorOperation::activation_func_in_situ(cur_input_gate, "sigmoid");
		
		TensorOperation::matmul(cur_input, forget_gate_input_weight,
				input_product.get_var_data());
                TensorOperation::matmul(last_output, forget_gate_output_weight,
				output_product.get_var_data());
                TensorOperation::add(input_product, output_product,
				cur_forget_gate.get_var_data());
		cur_forget_gate += forget_gate_bias;
		cur_forget_gate += forget_gate_bias_const;
                TensorOperation::activation_func_in_situ(cur_forget_gate, "sigmoid");

		TensorOperation::matmul(cur_input, output_gate_input_weight,
				input_product.get_var_data());
                TensorOperation::matmul(last_output, output_gate_output_weight,
				output_product.get_var_data());
                TensorOperation::add(input_product, output_product,
				cur_output_gate.get_var_data());
		cur_output_gate += output_gate_bias;
                TensorOperation::activation_func_in_situ(cur_output_gate, "sigmoid");

		TensorOperation::matmul(cur_input, state_input_weight,
				input_product.get_var_data());
                TensorOperation::matmul(last_output, state_output_weight,
				output_product.get_var_data());
                TensorOperation::add(input_product, output_product,
				cur_tmp_state.get_var_data());
                cur_tmp_state += state_bias;
                TensorOperation::activation_func_in_situ(cur_tmp_state, "tanh");

		TensorOperation::mul(last_state, cur_forget_gate,
				last_product.get_var_data());
		TensorOperation::mul(cur_tmp_state, cur_input_gate,
				new_product.get_var_data());

		TensorOperation::add(last_product, new_product,
				cur_state.get_var_data());
		
		cur_output.deep_copy(cur_state);
		TensorOperation::activation_func_in_situ(cur_output, "tanh");
		cur_output *= cur_output_gate;
		
		last_state.deep_copy(cur_state);
		last_output.deep_copy(cur_output);
	}
	return output;
}
