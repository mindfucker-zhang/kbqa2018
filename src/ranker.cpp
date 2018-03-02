#include "ranker.h"
#include "utils.h"
#include "logger.h"

liner_ranker_dict* LinerRanker::global_init(const std::string& path){
	liner_ranker_dict* dict = new liner_ranker_dict();
	if (Utils::load_dict(path, dict->weights, ": ") != 0){
		Logger::logging("LinerRanker global init error: " + path, "ERROR");
		delete dict;
		return NULL;
	}
	return dict;
}

void LinerRanker::global_destroy(void*& dict){
	if (dict){
		liner_ranker_dict* ldict = (liner_ranker_dict*) dict;
		delete ldict;
		dict = NULL;
	}
}

int LinerRanker::thread_init(void* dict){
		return 0;
}

void LinerRanker::thread_destroy(){}

void LinerRanker::clear(){}

int LinerRanker::scoring(const std::map<std::string, float>& feats, float& score, void* dict){
	liner_ranker_dict* liner_dict = (liner_ranker_dict*) dict;
	score = 0.0;
	for (auto iter = liner_dict->weights.begin(); iter != liner_dict->weights.end(); ++iter){
		auto fiter = feats.find(iter->first);
		if (fiter == feats.end()){
			score = 0.0;
			Logger::logging("cannot get feature: " + iter->first, "ERROR");
			return -1;
		}
		score += iter->second * fiter->second;
	}
	return 0;
}

