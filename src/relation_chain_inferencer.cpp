#include "relation_chain_inferencer.h"
#include "utils.h"
#include "logger.h"

relation_chain_inferencer_dict* RelationChainInferencer::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
			Logger::logging("relation chain inferencer read conf map error!", "ERROR");
			return NULL;
	}
	relation_chain_inferencer_dict* dict = new relation_chain_inferencer_dict();

	char* keys[] = {"scorer", "ranker"};
	int key_num = 2;
	
	int ret = 0;
	
	std::vector<std::string> strs;
	std::vector<std::string> info; 
	// for ranker: info[0] is type , info[1] is dict path 
	// for scorers: info[0] is key, info[1] is type, info[2] is dict path
	
	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + "not in conf map!", "ERROR");
			ret = -1;
			break;
		}
		
		if (strcmp(keys[i], "ranker") == 0){
			Utils::split(iter->second, "|", info);
			if (info.size() < 2){
				ret = -1;
			}
			else{
				void* rdict = NULL;
				if (info[0] == "liner"){
					rdict = (void*) LinerRanker::global_init(info[1]);
				}//add else-if for other ranker 
				
				if (rdict){
					dict->ranker_type = info[0];
					dict->ranker_dict = rdict;
				}else{
					ret = -1;
				}
			}
			
		}else if (strcmp(keys[i], "scorer") == 0){
			if (iter->second =="NULL"){
				continue;
			}
			Utils::split(iter->second, ",", strs);
			for (int j = 0; j < strs.size(); ++j){
				Utils::split(strs[j], "|", info);
				if (info.size() < 3){
					ret = -1;
					break;
					
				}else{
					void* sdict = NULL;
					if (info[1] == "relation_chain_cnn"){
						sdict = (void*) RelationChainCNNScorer::global_init(info[2]);
					}// add else-if for other scorer 
					if (sdict){
						dict->scorer_type[info[0]] = info[1];
						dict->scorer_dict[info[0]] = sdict;
					}else{
						ret = -1;
						break;
					}
					
				}
			}
			
		}
		
		if (ret != 0){
			Logger::logging(std::string(keys[i]) + " load error!", "ERROR");
			break;
		}
	}
	if (ret != 0){
		global_destroy(dict);
		return NULL;
	}
	return dict;
}

void RelationChainInferencer::global_destroy(relation_chain_inferencer_dict*& dict){
	if (!dict){
		return;
	}
	if (dict->ranker_dict){
		if (dict->ranker_type == "liner"){
			LinerRanker::global_destroy(dict->ranker_dict);
		}
	}
	
	for (auto iter = dict->scorer_dict.begin(); 
			iter != dict->scorer_dict.end(); ++iter){
		if (iter->second){
			std::string type = dict->scorer_type[iter->first];
			if (type == "relation_chain_cnn"){
				RelationChainCNNScorer::global_destroy(iter->second);
			}
		}
	}
	delete dict;
	dict = NULL;
}

int RelationChainInferencer::thread_init(relation_chain_inferencer_dict* dict){
	_ranker = NULL;
	if (dict->ranker_type == "liner"){
		_ranker = new LinerRanker();
	}
	if (!_ranker || _ranker->thread_init(dict->ranker_dict) != 0){
		return -1;
	}
	
	for (auto iter = dict->scorer_type.begin(); 
			iter != dict->scorer_type.end(); ++iter){
		_scorers[iter->first] = NULL;
		if (iter->second == "relation_chain_cnn"){
			_scorers[iter->first] = new RelationChainCNNScorer();
		}
		if (!_scorers[iter->first] || _scorers[iter->first]->thread_init(
				dict->scorer_dict[iter->first]) != 0){
			return -1;
		}
	}
	return 0;
}

void RelationChainInferencer::thread_destroy(){
	if (_ranker){
		_ranker->thread_destroy();
		delete _ranker;
		_ranker = NULL;
	}
	for (auto iter = _scorers.begin(); iter != _scorers.end(); ++iter){
		if (iter->second){
			iter->second->thread_destroy();
			delete iter->second;
			iter->second = NULL;
		}
	}
	_scorers.clear();
}

int RelationChainInferencer::char_based_f1_bleu_feature(Question& q, const std::string& key){
	const int const_max_ngram = 2;
	std::map<cand_relation_chain_key, cand_relation_chain_node*>& cand_relation_chains = q.cands->cand_relation_chains;
	std::map<std::string, std::vector<std::map<std::string, int> > > pat_char_map;
	std::map<std::string, std::vector<std::map<std::string, int> > > rc_char_map;
	
	for (auto kv = cand_relation_chains.begin(); kv != cand_relation_chains.end(); ++kv){
		cand_relation_chain_node* cand = kv->second;
		const std::vector<std::string>& p_char_vec = cand->topic_mention->pat_char_vec;
		int r_char_vec_size = 0;
		for (int i = 0; i < cand->relations.size(); ++i){
			r_char_vec_size += cand->relations[i]->char_vec.size();
		}
		
		const std::string& pat = cand->topic_mention->pat;
		const std::string& rc = cand->str;
		if (pat_char_map.find(pat) == pat_char_map.end()){
			int max_ngram = (p_char_vec.size() < const_max_ngram)? p_char_vec.size(): const_max_ngram;
			std::vector<std::map<std::string, int> > rec;
			std::string ngram_char;
			rec.resize(max_ngram);
			for (int ngram = 1; ngram <= max_ngram; ++ngram){
				std::map<std::string, int>& cur_rec = rec[ngram - 1];
				for(int i = 0; i <= p_char_vec.size() - ngram; ++i){
					Utils::join(p_char_vec.begin() + i, p_char_vec.begin() + i + ngram, ngram_char);
					if (cur_rec.find(ngram_char) == cur_rec.end()){
						cur_rec[ngram_char] = 1; 
					}else{
						++cur_rec[ngram_char];
					}
				}
			}
			pat_char_map[pat] = std::move(rec);
		}
		
		if (rc_char_map.find(rc) == rc_char_map.end()){
			int max_ngram = (r_char_vec_size < const_max_ngram)? r_char_vec_size: const_max_ngram;
			std::vector<std::string> r_char_vec;
			for (int i = 0; i < cand->relations.size(); ++i){
				for (int j = 0; j < cand->relations[i]->char_vec.size(); ++j){
					r_char_vec.push_back(cand->relations[i]->char_vec[j]);
				}
			}
			std::vector<std::map<std::string, int> > rec;
			std::string ngram_char;
			rec.resize(max_ngram);
			for (int ngram = 1; ngram <= max_ngram; ++ngram){
				std::map<std::string, int>& cur_rec = rec[ngram - 1];
				for(int i = 0; i <= r_char_vec.size() - ngram; ++i){
					Utils::join(r_char_vec.begin() + i, r_char_vec.begin() + i + ngram, ngram_char);
					if (cur_rec.find(ngram_char) == cur_rec.end()){
						cur_rec[ngram_char] = 1; 
					}else{
						++cur_rec[ngram_char];
					}
				}
			}
			rc_char_map[rc] = std::move(rec);
		}
		
		std::vector<std::map<std::string, int> >& p_rec = pat_char_map[pat];
		std::vector<std::map<std::string, int> >& r_rec = rc_char_map[rc];

		/*std::cout << "pat:" << pat << "\n";
		std::cout << "osize:"  << p_rec.size() << "\n"; 
		for (auto iter = p_rec.begin(); iter != p_rec.end(); ++iter){
			std::cout << "isize: " << iter->size() << "\n";
			for (auto jter = iter->begin(); jter != iter->end(); ++jter){
				std::cout << jter->first << "\t" << jter->second << "\n";
			}
		}

		std::cout << "rc:" << rc << "\n";
                std::cout << "osize:"  << r_rec.size() << "\n";
                for (auto iter = r_rec.begin(); iter != r_rec.end(); ++iter){
                        std::cout << "isize: " << iter->size() << "\n";
                        for (auto jter = iter->begin(); jter != iter->end(); ++jter){
                                std::cout << jter->first << "\t" << jter->second << "\n";
                        }
                }*/

		
		int max_ngram = const_max_ngram;
		if (r_char_vec_size < max_ngram){
			max_ngram = r_char_vec_size;
		}
		if (p_char_vec.size() < max_ngram){
			max_ngram = p_char_vec.size();
		}
		float sum = 0.0;
		for (int ngram = 1; ngram <= max_ngram ; ++ngram){
			std::map<std::string, int>& p_cur_rec = p_rec[ngram - 1];
			std::map<std::string, int>& r_cur_rec = r_rec[ngram - 1];
			float overlap = 0.0;
			for (auto riter = r_cur_rec.begin(); riter != r_cur_rec.end(); ++riter){
				auto piter = p_cur_rec.find(riter->first);
				if (piter != p_cur_rec.end()){
					overlap += (riter->second < piter->second)? riter->second: piter->second;
				}
			}
			float recall = overlap / (p_char_vec.size() - ngram + 1);
			float precision = overlap / (r_char_vec_size - ngram + 1);
			if (recall + precision > 0.0000001){
				sum += 2 * recall + precision / (recall + precision); 
			}
		}
		
		cand->feats[key] = (max_ngram)? sum / max_ngram: 0; 
	}
}

int RelationChainInferencer::relation_chain_basic_feature_extract(Question& q){
	char_based_f1_bleu_feature(q, "relation_chain_bleu_score");
}

int RelationChainInferencer::process(Question& q, relation_chain_inferencer_dict* dict){
	relation_chain_basic_feature_extract(q);
	
	for (auto iter = _scorers.begin(); iter != _scorers.end(); ++iter){
		if (iter->second->scoring(q, dict->scorer_dict[iter->first], iter->first) != 0){
			Logger::logging(iter->first + " scoring error!", "ERROR");
			return -1;
		}
	}
	
	std::map<cand_relation_chain_key, cand_relation_chain_node*>& cand_relation_chains = q.cands->cand_relation_chains;
	std::vector<cand_answer_node*>& cand_answers = q.cands->cand_answers;
	
	for (auto iter = cand_relation_chains.begin(); iter != cand_relation_chains.end(); ++iter){
		if (_ranker->scoring(iter->second->feats, iter->second->score, dict->ranker_dict) != 0){
			Logger::logging("relation chain ranking scoring error!", "ERROR");
			return -1;
		}
	}
	
	for (int i = 0; i < cand_answers.size(); ++i){
		cand_answers[i]->feats["relation_chain_score"] = cand_answers[i]->relation_chain->score;
		for (auto iter = cand_answers[i]->relation_chain->feats.begin(); 
				iter != cand_answers[i]->relation_chain->feats.end(); ++iter){
			cand_answers[i]->feats[iter->first] = iter->second;
		}
	}
	
	return 0;
}
