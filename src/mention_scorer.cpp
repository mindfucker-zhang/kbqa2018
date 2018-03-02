#include <math.h>
#include "mention_scorer.h"
#include "utils.h"
#include "logger.h"

mention_net_scorer_dict* MentionNetScorer::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
			Logger::logging("MentionNetScorer read conf map error!", "ERROR");
			return NULL;
	}
	mention_net_scorer_dict* dict = new mention_net_scorer_dict();
	char* keys[] = {"net_dict_conf", "feed_conf"};
	int key_num = 2;
	int ret = 0;
	
	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + "not in conf map!", "ERROR");
			ret = -1;
			break;
		}
		if (strcmp(keys[i], "net_dict_conf") == 0){
			dict->mention_net_dict = Net::global_init(iter->second);
			if (!dict->mention_net_dict){
				ret = -1;
			}
		}else if (strcmp(keys[i], "feed_conf") == 0){
			ret = Utils::read_conf_map(iter->second, dict->feed_conf);
			if (ret == 0){
				ret = WordSeqDataFeed::global_feed_dict_init(dict->feed_conf, dict->feed_dict);
			}
		}
		
		if (ret != 0){
			Logger::logging(std::string(keys[i]) + " load error!", "ERROR");
			break;
		}
	}
	
	if (ret != 0){
		Net::global_destroy(dict->mention_net_dict);
		WordSeqDataFeed::global_feed_dict_destroy(dict->feed_dict);
		delete dict;
		return NULL;
	}
	return dict;
}

void MentionNetScorer::global_destroy(void*& dict){
	if (dict){
		mention_net_scorer_dict* mdict = (mention_net_scorer_dict*) dict;
		Net::global_destroy(mdict->mention_net_dict);
		WordSeqDataFeed::global_feed_dict_destroy(mdict->feed_dict);
		delete mdict;
		dict = NULL;
	}
}

int MentionNetScorer::thread_init(void* dict){
	mention_net_scorer_dict* mdict = (mention_net_scorer_dict*) dict;
	if (_net.thread_init(mdict->mention_net_dict->para_conf, 
			mdict->mention_net_dict->batch_size) != 0){
		Logger::logging("mention net thread init error!", "ERROR");
		return -1;
	}
	if (_data_feed.thread_init(mdict->feed_conf) != 0){
		Logger::logging("mention data feed thread init error!", "ERROR");
		return -1;
	}
	return 0;
}

void MentionNetScorer::thread_destroy(){
	_net.thread_destroy();
	_data_feed.thread_destroy();
}

void MentionNetScorer::clear(){}

int MentionNetScorer::scoring(Question &q, void* dict, const std::string& key){
	mention_net_scorer_dict* mdict = (mention_net_scorer_dict*) dict;
	int word_seq_len = Utils::str2int(mdict->feed_conf["word_seq_len"]);
	_input.words = q.word_seg;
	std::map<std::string, Tensor>& input = _data_feed.gen_input(
			(void*) &_input, (void*) &(mdict->feed_dict));

	const std::map<std::string, Tensor>& net_output = _net.run(
			input, mdict->mention_net_dict->model.params);
	auto iter = net_output.find("probs");
	const DataType* probs = iter->second.get_data();
	
	std::vector<cand_mention_node*>& cand_mentions = q.cands->cand_mentions;
	for (int i = 0; i < cand_mentions.size(); ++i){
		float score = 0.0;
		if (cand_mentions[i]->word_en_index < word_seq_len - 1){
			score = probs[cand_mentions[i]->word_st_index * 3 ] * 
					probs[(cand_mentions[i]->word_en_index + 1) * 3 + 1];
			score = -1.0 / log(score);
		}
		//std::cout << cand_mentions[i]->str << "\t" << cand_mentions[i]->word_st_index << "\t"
		//	<< cand_mentions[i]->word_en_index << "\t" << score << "\n";
		cand_mentions[i]->feats[key] = score; 
	}
	
	return 0;
}
