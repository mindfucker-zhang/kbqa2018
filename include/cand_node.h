#ifndef _CAND_NODE_H_
#define _CAND_NODE_H_
#include <map>
#include <vector>
#include <string>

struct cand_mention_node{
	int id;
	std::string str;
	int word_st_index;
	int word_en_index;

	std::string pat;

	std::vector<std::string> pat_word_seg;
	std::vector<std::vector<std::string> > pat_char_seg;
	int pat_word_index;

	std::vector<std::string> pat_char_vec;
	int pat_char_index;

	std::map<std::string, float> feats;
	float score;
	
};

struct cand_entity_node{
	int id;
	std::string str;
	std::string mention;
	
	bool is_topic;
	cand_mention_node* topic_mention;
	std::map<std::string, float> topic_feats;
	float topic_score;
};

struct cand_relation_node{
	std::string str;
	std::vector<std::string> word_seg;
	std::vector<std::vector<std::string> > char_seg;
	std::vector<std::string> char_vec;
};

struct cand_relation_chain_node{
	std::string str;
	cand_mention_node* topic_mention;
	std::vector<cand_relation_node*> relations;
	
	std::map<std::string, float> feats;
	float score;
};

struct cand_answer_node{
	std::string str;
	cand_mention_node* topic_mention;
	cand_entity_node* topic_entity;
	cand_relation_chain_node* relation_chain; 
	std::vector<std::string> mentions;
	std::vector<cand_entity_node*> entities;

	std::map<std::string, float> feats;
	float score;
};

struct cand_topic_entity_key{
	int st;
	int en;
	int eid;
	
	bool operator<(const cand_topic_entity_key& key) const{
		if (st != key.st){
			return st < key.st;
		}
		if (en != key.en){
			return en < key.en;
		}
		return eid < key.eid;
	}
};

struct cand_relation_chain_key{
	int men_st;
	int men_en;
	std::string str;
	
	bool operator<(const cand_relation_chain_key& key) const{
		if (men_st != key.men_st){
			return men_st < key.men_st;
		}
		if (men_en != key.men_en){
			return men_en < key.men_en;
		}
		return str < key.str;
	}
};

struct cand_info{
	std::vector<cand_mention_node*>& cand_mentions;
	std::vector<cand_entity_node*>& cand_topic_entities; 
	std::vector<cand_entity_node*>& cand_entities;
	std::map<std::string, cand_relation_node*>& cand_relations;
	std::map<cand_relation_chain_key, cand_relation_chain_node*>& cand_relation_chains;
	std::vector<cand_answer_node*>& cand_answers;
	
	cand_info(std::vector<cand_mention_node*>& cmentions, 
			std::vector<cand_entity_node*>& ctopic_entities,
			std::vector<cand_entity_node*>& centities,
			std::map<std::string, cand_relation_node*>& crelations,
			std::map<cand_relation_chain_key, cand_relation_chain_node*>& crelation_chains,
			std::vector<cand_answer_node*>& canswers):
			cand_mentions(cmentions), cand_topic_entities(ctopic_entities), 
			cand_entities(centities), cand_relations(crelations),
			cand_relation_chains(crelation_chains), cand_answers(canswers){}
	
};

#endif
