#include <cstring>
#include <utility>
#include "relation_chain_scorer.h"
#include "utils.h"
#include "logger.h"
#include "sse_op.h"

relation_cnn_scorer_dict* RelationChainCNNScorer::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
		Logger::logging("RelationNetScorer read conf map error!", "ERROR");
		return NULL;
	}
	
	relation_cnn_scorer_dict* dict = new relation_cnn_scorer_dict();
	char* keys[] = {"sent_net_dict_conf", "rc_net_dict_conf", "sent_feed_conf", "rc_feed_conf"};
	int key_num = 4;
	int ret = 0;

	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + "not in conf map!", "ERROR");
			ret = -1;
			break;
		}
		if (strcmp(keys[i], "sent_net_dict_conf") == 0){
			dict->sent_cnn_net_dict = Net::global_init(iter->second);
			if (!dict->sent_cnn_net_dict){
				ret = -1;
			}
		}else if (strcmp(keys[i], "rc_net_dict_conf") == 0){
			dict->rc_cnn_net_dict = Net::global_init(iter->second);
			if (!dict->rc_cnn_net_dict){
				ret = -1;
			}
		}else if (strcmp(keys[i], "sent_feed_conf") == 0){
			ret = Utils::read_conf_map(iter->second, dict->sent_feed_conf);
			if (ret == 0){
				ret = WordCharSeqDataFeed::global_feed_dict_init(dict->sent_feed_conf, dict->feed_dict);
			}
		}else if (strcmp(keys[i], "rc_feed_conf") == 0){
			ret = Utils::read_conf_map(iter->second, dict->rc_feed_conf);
		}
		if (ret != 0){
			Logger::logging(std::string(keys[i]) + " load error!", "ERROR");
			break;
		}
	
	}
	
	if (ret != 0){
		void* vdict =(void*)dict;
		global_destroy(vdict);
		delete dict;
		return NULL;
	}
	return dict;
}

void RelationChainCNNScorer::global_destroy(void*& dict){
	if (!dict){
		return;
	}
	relation_cnn_scorer_dict* rdict = (relation_cnn_scorer_dict*) dict;
	Net::global_destroy(rdict->sent_cnn_net_dict);
	Net::global_destroy(rdict->rc_cnn_net_dict);
	WordCharSeqDataFeed::global_feed_dict_destroy(rdict->feed_dict);
	delete rdict;
	dict = NULL;
}

int RelationChainCNNScorer::thread_init(void* dict){
	relation_cnn_scorer_dict* rdict = (relation_cnn_scorer_dict*) dict;
	if (_sent_net.thread_init(rdict->sent_cnn_net_dict->para_conf,
			rdict->sent_cnn_net_dict->batch_size) != 0){
		Logger::logging("sent semantics embedding net thread init error!", "ERROR");
		return -1;
	}
	if (_sent_data_feed.thread_init(rdict->sent_feed_conf) != 0){
		Logger::logging("sent semantics embedding data feed thread init error!", "ERROR");
		return -1;
	}
	if (_rc_net.thread_init(rdict->rc_cnn_net_dict->para_conf,
			rdict->rc_cnn_net_dict->batch_size) != 0){
		Logger::logging("rc semantics embedding net thread init error!", "ERROR");
		return -1;
	}
	if (_rc_data_feed.thread_init(rdict->rc_feed_conf) != 0){
		Logger::logging("rc semantics embedding data feed thread init error!", "ERROR");
		return -1;
	}
		
	_sent_batch_size = Utils::str2int(rdict->sent_feed_conf["batch_size"]);
	_rc_batch_size = Utils::str2int(rdict->rc_feed_conf["batch_size"]);
	_sent_inputs.resize(_sent_batch_size);
	_rc_inputs.resize(_rc_batch_size);
	return 0;
}

void RelationChainCNNScorer::thread_destroy(){
	_sent_net.thread_destroy();
	_rc_net.thread_destroy();
	_sent_data_feed.thread_destroy();
	_rc_data_feed.thread_destroy();
	_sent_inputs.clear();
	_rc_inputs.clear();
}

void RelationChainCNNScorer::clear(){}

int RelationChainCNNScorer::scoring(Question &q, void* dict, const std::string& key){
	relation_cnn_scorer_dict* rdict = (relation_cnn_scorer_dict*) dict;
	int feat_num = _sent_net.get_feat_size();
	std::map<std::string, int> sent2id;
	std::map<std::string, int> rc2id;
	std::vector<cand_relation_chain_node*> id2rc;
	std::vector<void*> void_inputs;
	std::vector<cand_mention_node*>& mentions = q.cands->cand_mentions;
	std::map<cand_relation_chain_key, cand_relation_chain_node*>& rcs = q.cands->cand_relation_chains;
	DataType* sent_feats = NULL;
	DataType* rc_feats = NULL;

	sent_feats = new DataType[feat_num * mentions.size()];
	for (int i = 0; i < _sent_batch_size; ++i){
		void_inputs.push_back((void*) &(_sent_inputs[i]));
	}
	for (int i = 0; i < mentions.size(); ++i){
		sent2id[mentions[i]->pat] = i;
		int index = i % _sent_batch_size;
		_sent_inputs[index].words = mentions[i]->pat_word_seg;
		_sent_inputs[index].chars = mentions[i]->pat_char_seg;
		if (index == _sent_batch_size - 1){
			std::vector<std::map<std::string, Tensor> >& inputs = _sent_data_feed.gen_batch_input(
					void_inputs, (void*) &rdict->feed_dict);
			const std::vector<std::map<std::string, Tensor> >& outputs = _sent_net.run_batch(inputs,
					rdict->sent_cnn_net_dict->model.params);
			
			for (int j = i - _sent_batch_size + 1, k = 0; j <= i;  ++j, ++k){
				memcpy(sent_feats + j * feat_num, outputs[k].find("sem_feats")
						->second.get_data(), feat_num * sizeof(DataType));
			}
		}
	}
	int last_size = mentions.size() % _sent_batch_size;
	if (last_size){
		std::vector<std::map<std::string, Tensor> >& inputs = _sent_data_feed.gen_batch_input(
				void_inputs, (void*) &rdict->feed_dict);
		const std::vector<std::map<std::string, Tensor> >& outputs = _sent_net.run_batch(inputs,
				rdict->sent_cnn_net_dict->model.params);
		for (int j = mentions.size() - last_size, k = 0; k < last_size;  ++j, ++k){
			memcpy(sent_feats + j * feat_num, outputs[k].find("sem_feats")
					->second.get_data(), feat_num * sizeof(DataType));
		}
	}

	int rc_num = 0;
	for (auto iter = rcs.begin(); iter != rcs.end(); ++iter){
		cand_relation_chain_node* rc_node = iter->second;
		if (rc2id.find(rc_node->str) == rc2id.end()){
			rc2id[rc_node->str] = rc_num++;
			id2rc.push_back(rc_node);
		}
	}

	rc_feats = new DataType[feat_num * rc_num];
	void_inputs.clear();
	for (int i = 0; i < _rc_batch_size; ++i){
		void_inputs.push_back((void*) &(_rc_inputs[i]));
	}
	for (int i = 0; i < rc_num; ++i){
		int index = i % _rc_batch_size;
		gen_seq_input(id2rc[i], _rc_inputs[index]);
		if (index == _rc_batch_size - 1){
			std::vector<std::map<std::string, Tensor> >& inputs = _rc_data_feed.gen_batch_input(
					void_inputs, (void*) &rdict->feed_dict);
			const std::vector<std::map<std::string, Tensor> >& outputs = _rc_net.run_batch(inputs,
					rdict->rc_cnn_net_dict->model.params);

			for (int j = i - _rc_batch_size + 1, k = 0; j <= i;  ++j, ++k){
				memcpy(rc_feats + j * feat_num, outputs[k].find("sem_feats")
						->second.get_data(), feat_num * sizeof(DataType));
			}
		}
	}
	last_size = rc_num % _rc_batch_size;
	if (last_size){
		std::vector<std::map<std::string, Tensor> >& inputs = _rc_data_feed.gen_batch_input(
				void_inputs, (void*) &rdict->feed_dict);
		const std::vector<std::map<std::string, Tensor> >& outputs = _rc_net.run_batch(inputs,
				rdict->rc_cnn_net_dict->model.params);
		for (int j = rc_num - last_size, k = 0; k < last_size;  ++j, ++k){
			memcpy(rc_feats + j * feat_num, outputs[k].find("sem_feats")
					->second.get_data() ,feat_num * sizeof(DataType));
		}
	}

	std::map<std::pair<int, int>, DataType> sim_map;
	for (auto iter = rcs.begin(); iter != rcs.end(); ++iter){
		int sid = sent2id[iter->second->topic_mention->pat];
		int rid = rc2id[iter->second->str];
		std::pair<int, int> sim_key = std::make_pair(sid, rid);
		auto sim_iter = sim_map.find(sim_key);
		if (sim_map.find(sim_key) != sim_map.end()){
			iter->second->feats[key] = sim_iter->second;
		}else{
			DataType score = 0.0;
			sse_vector_dot_mul(sent_feats + feat_num * sid, rc_feats + feat_num * rid, feat_num, score);
			score /= 3000;
			sim_map[sim_key] = score;
			iter->second->feats[key] = score;
		}
	}
	delete[] sent_feats;
	delete[] rc_feats;
	return 0;
}

void RelationChainCNNScorer::gen_seq_input(cand_relation_chain_node* cand,
		word_char_seq_input& seq_input){
	seq_input.words.clear();
	seq_input.chars.clear();
	for (int i = 0; i < cand->relations.size(); ++i){
		cand_relation_node* node = cand->relations[i];
		for (int j = 0; j < node->word_seg.size(); ++j){
			seq_input.words.push_back(node->word_seg[j]);
			seq_input.chars.push_back(node->char_seg[j]);
		}
	}
}
