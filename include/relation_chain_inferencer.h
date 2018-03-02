#ifndef _RELATION_CHAIN_INFERENCER_H_
#define _RELATION_CHAIN_INFERENCER_H_
#include <map>
#include <string>
#include "question.h"
#include "ranker.h"
#include "relation_chain_scorer.h"

struct relation_chain_inferencer_dict{
	std::map<std::string, std::string> scorer_type;
	std::map<std::string, void*> scorer_dict;
	std::string ranker_type;
	void* ranker_dict;
};

class RelationChainInferencer{
	
public:
	int thread_init(relation_chain_inferencer_dict* dict);
	void thread_destroy();
	int process(Question& q, relation_chain_inferencer_dict* dict);
	
private:
	int relation_chain_basic_feature_extract(Question& q);
	int char_based_f1_bleu_feature(Question& q, const std::string& key);
	
private:
	Ranker* _ranker;
	std::map<std::string, RelationChainScorer*> _scorers;

public:
	static relation_chain_inferencer_dict* global_init(const std::string& conf_path);
	static void global_destroy(relation_chain_inferencer_dict*& dict);
};

#endif
