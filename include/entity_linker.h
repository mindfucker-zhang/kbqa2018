#ifndef _ENTITY_LINKER_H_
#define _ENTITY_LINKER_H_
#include <map>
#include <string>
#include "question.h"
#include "ranker.h"
#include "mention_scorer.h"
#include "entity_scorer.h"

struct entity_linker_dict{
	std::map<std::string, std::string> mention_scorer_type;
	std::map<std::string, void*> mention_scorer_dict;
	std::string mention_ranker_type;
	void* mention_ranker_dict;
	
	std::map<std::string, std::string> _entity_scorer_types;
	std::map<std::string, void*> _entity_scorer_dicts;
	std::string entity_ranker_type;
	void* entity_ranker_dict;
};

class EntityLinker{
	
public:
	int thread_init(entity_linker_dict* dict);
	void thread_destroy();
	int process(Question& q, entity_linker_dict* dict);

private:
	int mention_basic_feature_extract(Question& q);
	int entity_basic_feature_extract(Question& q);
	
private:
	Ranker* _mention_ranker;
	std::map<std::string, MentionScorer*> _mention_scorers;
	Ranker* _entity_ranker;
	std::map<std::string, EntityScorer*> _entity_scorers;
	
public:
	static entity_linker_dict* global_init(const std::string& conf_path);
	static void global_destroy(entity_linker_dict*& dict);
};

#endif
