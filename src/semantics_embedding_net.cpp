#include <iostream>
#include "semantics_embedding_net.h"
#include "concat_layer.h"
#include "dense_layer.h"
#include "embedding_layer.h"
#include "activate_layer.h"
#include "batch_normalization_layer.h"
#include "conv1d_layer.h"
#include "max1_pooling_layer.h"

const std::map<std::string, Tensor>& SemanticsEmbeddingNet::run(
  		std::map<std::string,Tensor>& input,
  		std::map<std::string, std::map<std::string, Tensor> >& param_map, 
		int index){

	std::map<std::string, Tensor> layer_input;

	layer_input["input"] = input["word_ids"];

	Layer* word_emb_layer = _layer_maps[index]["word_embedding"];	
	Tensor word_emb = word_emb_layer->process(layer_input, param_map["word_embedding"]);
	
	layer_input["input"] = input["char_ids"];
	Layer* char_emb_layer = _layer_maps[index]["char_embedding"];
	Tensor char_emb = char_emb_layer->process(layer_input, param_map["char_embedding"]);
	
	const std::vector<int>& char_emb_shape = char_emb.get_shape();
	int char_emb_dim = char_emb_shape.size();
	std::vector<int> char_feat_shape;
	for (int i = 0; i < char_emb_dim - 2; ++i){
		char_feat_shape.push_back(char_emb_shape[i]);
	}
	char_feat_shape.push_back(char_emb_shape[char_emb_dim - 2] 
			* char_emb_shape[char_emb_dim - 1]);
	char_emb.reshape(char_feat_shape);

	layer_input.clear();
	layer_input["input1"] = word_emb;
	layer_input["input2"] = char_emb;
	Layer* input_concat_layer = _layer_maps[index]["input_concat"];
	Tensor input_concat_feats = input_concat_layer->process(layer_input);

	layer_input.clear();
	layer_input["input"] = input_concat_feats;
	Layer* win2_conv1d = _layer_maps[index]["win2_conv1d"];
	((Conv1dLayer*) win2_conv1d)->set_bias_flag(false);
	Tensor win2_conv_feats = win2_conv1d->process(layer_input, param_map["win2_conv1d"]);
	Layer* win3_conv1d = _layer_maps[index]["win3_conv1d"];
	((Conv1dLayer*) win3_conv1d)->set_bias_flag(false);
	Tensor win3_conv_feats = win3_conv1d->process(layer_input, param_map["win3_conv1d"]);
	
	layer_input["input"] = win2_conv_feats;
	Layer* win2_bn = _layer_maps[index]["win2_bn"]; 
	Tensor win2_bn_feats = win2_bn->process(layer_input, param_map["win2_bn"]);
	
	layer_input["input"] = win3_conv_feats;
	Layer* win3_bn = _layer_maps[index]["win3_bn"];
	Tensor win3_bn_feats = win3_bn->process(layer_input, param_map["win3_bn"]);

	Layer* relu_activate =  _layer_maps[index]["relu_activate"];
	((ActivateLayer*) relu_activate)->set_func_key("relu");

	layer_input["input"] = win2_bn_feats;
	Tensor win2_relu_feats = relu_activate->process(layer_input);
	layer_input["input"] = win3_bn_feats;
	Tensor win3_relu_feats = relu_activate->process(layer_input);

	layer_input["input"] = win2_relu_feats;
	Layer* win2_max1pool = _layer_maps[index]["win2_max1pool"];
	Tensor win2_pool_feats = win2_max1pool->process(layer_input);

	layer_input["input"] = win3_relu_feats;
	Layer* win3_max1pool = _layer_maps[index]["win3_max1pool"];
	Tensor win3_pool_feats = win3_max1pool->process(layer_input);

	layer_input.clear();
	layer_input["input1"] = win2_pool_feats;
	layer_input["input2"] = win3_pool_feats;
	Layer* feat_concat_layer = _layer_maps[index]["feat_concat"];
	Tensor concat_feat = feat_concat_layer->process(layer_input);

	layer_input.clear();
	layer_input["input"] = concat_feat;
	Layer* dense_layer = _layer_maps[index]["dense"];
	((DenseLayer*) dense_layer)->set_bias_flag(false);
	Tensor dense_feat = dense_layer->process(layer_input, param_map["dense"]);
	
	layer_input["input"] = dense_feat;
	Layer* feat_bn_layer = _layer_maps[index]["feat_bn"];
	Tensor bn_feats = feat_bn_layer->process(layer_input, param_map["feat_bn"]);

	layer_input["input"] = bn_feats;
	Tensor sem_feats = relu_activate->process(layer_input);
	
	_batch_results[index]["sem_feats"] = sem_feats;
	return _batch_results[index];
}

int SemanticsEmbeddingNet::get_feat_size(){
	if (!_batch_size){
		return -1;
	}
	std::vector<int> feat_shape;
	_layer_maps[0]["feat_bn"]->get_tensor_shape("output", feat_shape);
	if (!feat_shape.size()){
		return -1;
	}
	return feat_shape[feat_shape.size() - 1];
}
