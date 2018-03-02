#include "entity_linker.h"
#include "utils.h"
#include "logger.h"

entity_linker_dict* EntityLinker::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
			Logger::logging("entity linker read conf map error!", "ERROR");
			return NULL;
	}
	entity_linker_dict* dict = new entity_linker_dict();

	/* feature work
	char* keys[] = {"mention_scorer", "mention_ranker", "entity_scorer", "entity_ranker"};
	int key_num = 4; */
	char* keys[] = {"mention_scorer", "mention_ranker"};
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
		
		if (strcmp(keys[i], "mention_ranker") == 0){
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
					dict->mention_ranker_type = info[0];
					dict->mention_ranker_dict = rdict;
				}else{
					ret = -1;
				}
			}
			
		}else if (strcmp(keys[i], "mention_scorer") == 0){
			if (iter->second == "NULL"){
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
					if (info[1] == "mention_net"){
						sdict = (void*) MentionNetScorer::global_init(info[2]);
					}// add else-if for other scorer 
					if (sdict){
						dict->mention_scorer_type[info[0]] = info[1];
						dict->mention_scorer_dict[info[0]] = sdict;
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

void EntityLinker::global_destroy(entity_linker_dict*& dict){
	if (!dict){
		return;
	}
	if (dict->mention_ranker_dict){
		if (dict->mention_ranker_type == "liner"){
			LinerRanker::global_destroy(dict->mention_ranker_dict);
		}
	}
	
	for (auto iter = dict->mention_scorer_dict.begin(); 
			iter != dict->mention_scorer_dict.end(); ++iter){
		if (iter->second){
			std::string type = dict->mention_scorer_type[iter->first];
			if (type == "mention_net"){
				MentionNetScorer::global_destroy(iter->second);
			}
		}
	}
	delete dict;
	dict = NULL;
}

int EntityLinker::thread_init(entity_linker_dict *dict){
	_mention_ranker = NULL;
	if (dict->mention_ranker_type == "liner"){
		_mention_ranker = new LinerRanker();
	}
	if (!_mention_ranker || _mention_ranker->thread_init(dict->mention_ranker_dict) != 0){
		return -1;
	}
	
	for (auto iter = dict->mention_scorer_type.begin(); 
			iter != dict->mention_scorer_type.end(); ++iter){
		_mention_scorers[iter->first] = NULL;
		if (iter->second == "mention_net"){
			_mention_scorers[iter->first] = new MentionNetScorer();
		}
		if (!_mention_scorers[iter->first] || _mention_scorers[iter->first]->thread_init(
				dict->mention_scorer_dict[iter->first]) != 0){
			return -1;
		}
	}
	return 0;
}

void EntityLinker::thread_destroy(){
	if (_mention_ranker){
		_mention_ranker->thread_destroy();
		delete _mention_ranker;
		_mention_ranker = NULL;
	}
	for (auto iter = _mention_scorers.begin(); iter != _mention_scorers.end(); ++iter){
		if (iter->second){
			iter->second->thread_destroy();
			delete iter->second;
			iter->second = NULL;
		}
	}
	_mention_scorers.clear();
}

int EntityLinker::mention_basic_feature_extract(Question& q){
	
}

int EntityLinker::entity_basic_feature_extract(Question& q){
	
}


int EntityLinker::process(Question& q, entity_linker_dict* dict){
	mention_basic_feature_extract(q);
	
	for (auto iter = _mention_scorers.begin(); iter != _mention_scorers.end(); ++iter){
		if (iter->second->scoring(q, dict->mention_scorer_dict[iter->first], iter->first) != 0){
			Logger::logging(iter->first + " scoring error!", "ERROR");
			return -1;
		}
	}

	entity_basic_feature_extract(q);
	
	std::vector<cand_mention_node*>& cand_mentions = q.cands->cand_mentions;
	std::vector<cand_answer_node*>& cand_answers = q.cands->cand_answers;
	
	for (int i = 0; i < cand_mentions.size(); ++i){
		if (_mention_ranker->scoring(cand_mentions[i]->feats,
				cand_mentions[i]->score, dict->mention_ranker_dict) != 0){
			Logger::logging("mention ranking scoring error!", "ERROR");
			return -1;
		}
	}
	
	for (int i = 0; i < cand_answers.size(); ++i){
		cand_answers[i]->feats["mention_score"] = cand_answers[i]->topic_mention->score;
		for (auto iter = cand_answers[i]->topic_mention->feats.begin(); 
				iter != cand_answers[i]->topic_mention->feats.end(); ++iter){
			cand_answers[i]->feats[iter->first] = iter->second;
		}
	}

	return 0;
}
