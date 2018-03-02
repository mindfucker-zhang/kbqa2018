#include "mention_rank_net.h"
#include "dense_layer.h"
#include "embedding_layer.h"
#include "lstm_layer.h"
#include "softmax_layer.h"
#include "concat_layer.h"

const std::map<std::string, Tensor>& MentionRankNet::run(std::map<std::string,Tensor>& input, 
		std::map<std::string, std::map<std::string, Tensor> >& param_map, int index){
	std::map<std::string, Tensor> layer_input;
	layer_input["input"] = input["word_ids"];
	Layer* word_emb_layer = _layer_maps[index]["word_embedding"];
	Tensor word_emb = word_emb_layer->process(layer_input, param_map["word_embedding"]);

	layer_input["input"] = word_emb;
	Layer* f_lstm_layer = _layer_maps[index]["f_lstm"];
	Layer* b_lstm_layer = _layer_maps[index]["b_lstm"];
	((LstmLayer*) f_lstm_layer)->set_reversed(false);
	((LstmLayer*) b_lstm_layer)->set_reversed(true);
	Tensor f_lstm_feats = f_lstm_layer->process(layer_input, param_map["f_lstm"]);
	Tensor b_lstm_feats = b_lstm_layer->process(layer_input, param_map["b_lstm"]);
	
	layer_input.clear();
	layer_input["input1"] = f_lstm_feats;
	layer_input["input2"] = b_lstm_feats;
	Layer* concat_layer = _layer_maps[index]["feat_concat"];
	Tensor concat_feats = concat_layer->process(layer_input);
	
	layer_input.clear();
	layer_input["input"] = concat_feats;
	Layer* dense_layer = _layer_maps[index]["dense"];
	Tensor final_feats = dense_layer->process(layer_input, param_map["dense"]);

	layer_input["input"] = final_feats;
	Layer* softmax_layer = _layer_maps[index]["softmax"];
	Tensor probs = softmax_layer->process(layer_input, param_map["softmax"]);

	_batch_results[index]["probs"] = probs;
	_batch_results[index]["emb"] = word_emb;
	_batch_results[index]["lstm"] = concat_feats;
	return _batch_results[index];
}
