#include "answer_reranker.h"
#include "utils.h"
#include "logger.h"

answer_reranker_dict*  AnswerReranker::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
		Logger::logging("answer reranker read conf map error!", "ERROR");
		return NULL;
	}
	answer_reranker_dict* dict = new answer_reranker_dict();
	char* keys[] = {"ranker"};
	int key_num = 1;
	int ret = 0;
	std::vector<std::string> info;

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

void AnswerReranker::global_destroy(answer_reranker_dict*& dict){
	if (!dict){
		return;
	}
	if (dict->ranker_dict){
		if (dict->ranker_type == "liner"){
			LinerRanker::global_destroy(dict->ranker_dict);
		}
	}
	delete dict;
	dict = NULL;
}

int AnswerReranker::thread_init(answer_reranker_dict* dict){
	_ranker = NULL;
	if (dict->ranker_type == "liner"){
		_ranker = new LinerRanker();
	}
	if (!_ranker || _ranker->thread_init(dict->ranker_dict) != 0){
		return -1;
	}
	return 0;
}

void AnswerReranker::thread_destroy(){
	if (_ranker){
		_ranker->thread_destroy();
		delete _ranker;
		_ranker = NULL;
	}
}

int AnswerReranker::scoring(Question& q, answer_reranker_dict* dict){
	std::vector<cand_answer_node*> cands = q.cands->cand_answers;
	for (int i = 0; i < cands.size(); ++i){
		if(_ranker->scoring(cands[i]->feats, cands[i]->score, dict->ranker_dict) != 0){
			Logger::logging("answer rearanking error!", "ERROR");
			return -1;
		}
	}
	return 0;
} 
