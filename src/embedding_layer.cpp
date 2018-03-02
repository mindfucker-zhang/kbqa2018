#include <cstring>
#include "embedding_layer.h"

int EmbeddingLayer::thread_init(){
	const std::vector<int>& output_shape = _param_conf["output_shape"];
	int size = 1;
	for (int i = 0; i < output_shape.size(); ++i){
		size *= output_shape[i];
	}
	_memory_map["output"] = new DataType[size];
	_tensor_map["output"] = Tensor(output_shape, false, _memory_map["output"]);
	return 0;
}

Tensor EmbeddingLayer::process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params){
	Tensor& input_tensor = inputs["input"];
	Tensor& embeddings = params["embedding"];

	const std::vector<int>& emb_shape = embeddings.get_shape();
	int emb_dim = emb_shape[1];
	int input_num = input_tensor.get_size();

	const DataType* indexes = input_tensor.get_data();
	DataType* data = _memory_map["output"];
	const DataType* emb_data = embeddings.get_data();
	for (int i = 0; i < input_num; ++i){
		int emb_pos = indexes[i] * emb_dim;
		int data_pos = i * emb_dim;
		memcpy(data + data_pos, emb_data + emb_pos, emb_dim * sizeof(DataType));
		/*for (int j = 0; j < emb_dim; ++ j){
			data[data_pos + j] = emb_data[emb_pos + j];
		}*/
	}
	return _tensor_map["output"];
}
