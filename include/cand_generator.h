#ifndef _CAND_GENERATOR_H_
#define _CAND_GENERATOR_H_
#include "preprocessor.h"
#include "question.h"
#include "cand_node.h"
#include "knowledge_base.h"

class CandGenerator{

public:
	int thread_init(Preprocessor* pre_thread, KnowledgeBase* kb_thread);
	void thread_destroy();
	void clear();
	int generate(Question& q);

private:
	cand_mention_node* generate_topic_mention(Question &q,
			int st, int en, const std::string& men_str);
	std::vector<cand_entity_node*> generate_entities_by_mention(const std::string& 
			men_str, bool is_topic = false);
	cand_relation_node* generate_cand_relation(const std::string& rel_str);
	cand_relation_chain_node* generate_cand_relation_chain(cand_entity_node* entity,
			const std::pair<std::string, std::string>& po, cand_relation_chain_node* father);
	cand_answer_node* generate_answer(cand_mention_node* mention, cand_entity_node* entity, 
			const std::pair<std::string, std::string>& po);
	cand_answer_node* generate_answer(cand_entity_node* entity, cand_answer_node* father,
			const std::pair<std::string, std::string>& po);
	int generate_cand_mentions(Question& q);
	int generate_cand_answers();

private:
	Preprocessor* _preprocess_thread;
	KnowledgeBase* _kb_thread;
	std::vector<cand_mention_node*> _cand_mentions;
	std::vector<cand_entity_node*> _cand_topic_entities; 
	std::vector<cand_entity_node*> _cand_entities;
	std::map<std::string, cand_relation_node*> _cand_relations;
	std::map<cand_relation_chain_key, cand_relation_chain_node*> _cand_relation_chains;
	std::vector<cand_answer_node*> _cand_answers;
	cand_mention_node* _tmp_father_mention;
private:	
	static std::map<std::string, float> _s_limits;

public:
	static int global_init(const std::string& conf_path);
	static void global_destroy();
};

#endif
