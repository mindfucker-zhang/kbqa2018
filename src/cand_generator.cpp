#include "cand_generator.h"
#include "utils.h"
#include "logger.h"

std::map<std::string, float> CandGenerator::_s_limits;

int CandGenerator::global_init(const std::string& conf_path){
	int ret = Utils::load_dict(conf_path, _s_limits, ": ");
	if (ret != 0){
		Logger::logging("cand_generator conf path error: " +  conf_path, "ERROR");
		_s_limits["relation_chain_max_len"] = 1;
		return -1;
	}
	if (_s_limits.find("relation_chain_max_len") == _s_limits.end() || 
			_s_limits["relation_chain_max_len"] < 1){
		_s_limits["relation_chain_max_len"] = 1;
	}
	return 0;
}

void CandGenerator::global_destroy(){
	_s_limits.clear();
}

int CandGenerator::thread_init(Preprocessor* pre_thread, KnowledgeBase* kb_thread){
	if (!pre_thread){
		Logger::logging("preprocess thread is null", "ERROR");
		return -1;
	}
	if (!kb_thread){
                Logger::logging("knowledge base thread is null", "ERROR");
                return -1;
        }

	_preprocess_thread = pre_thread;
	_kb_thread = kb_thread;
	return 0;
}

void CandGenerator::thread_destroy(){
	_preprocess_thread = NULL;
        _kb_thread = NULL;
	clear();
}

void CandGenerator::clear(){
	for (auto iter = _cand_mentions.begin(); iter != _cand_mentions.end(); ++iter){
		if (*iter){
			delete *iter;
		}
	}
	_cand_mentions.clear();
	
	_cand_topic_entities.clear();
	
	for (auto iter = _cand_entities.begin(); iter != _cand_entities.end(); ++iter){
                if (*iter){
                        delete *iter;
                }
        }
	_cand_entities.clear();
	
	for (auto iter = _cand_relations.begin(); iter != _cand_relations.end(); ++iter){
                if (iter->second){
                        delete iter->second;
                }
        }
        _cand_relations.clear();

	for (auto iter = _cand_relation_chains.begin(); iter !=  _cand_relation_chains.end(); ++iter){
                if (iter->second){
                        delete iter->second;
                }
        }
        _cand_relation_chains.clear();

	for (auto iter = _cand_answers.begin(); iter != _cand_answers.end(); ++iter){
                if (*iter){
                        delete *iter;
                }
        }
        _cand_answers.clear();

}

cand_mention_node* CandGenerator::generate_topic_mention(Question &q,
		int st, int en, const std::string& men_str){
	int id = _kb_thread->get_mention_id(men_str);
	if (id == -1){
		return NULL;
	}

	cand_mention_node* node = new cand_mention_node();
        node->id = id;
	node->str= men_str;
	node->word_st_index = st;
	node->word_en_index = en;
	node->pat = "";
	node->pat_word_index = st;
	node->pat_char_index = 0;
	for (int i = 0; i < st; ++i){
		node->pat += q.word_seg[i];
		node->pat_word_seg.push_back(q.word_seg[i]);
		node->pat_char_seg.push_back(q.char_seg[i]);
		node->pat_char_index += q.char_seg[i].size();
	}
	node->pat += "<entity>";
	node->pat_word_seg.push_back("<entity>");
	std::vector<std::string> tmp_char;
	tmp_char.push_back("<entity>");
	node->pat_char_seg.push_back(tmp_char);
	for (int i = en; i < q.word_seg.size(); ++i){
		node->pat += q.word_seg[i];
		node->pat_word_seg.push_back(q.word_seg[i]);
		node->pat_char_seg.push_back(q.char_seg[i]);
	}

	int rindex = node->pat_char_index;
	for (int i = st; i < en; ++i){
		 rindex += q.char_seg[i].size();
	}
	for (int i = 0; i < node->pat_char_index; ++ i){
		node->pat_char_vec.push_back(q.char_vec[i]);
	}
	node->pat_char_vec.push_back("<entity>");
	for (int i = rindex; i < q.char_vec.size(); ++i){
		node->pat_char_vec.push_back(q.char_vec[i]);
	}
	return node;
}

int CandGenerator::generate_cand_mentions(Question& q){
	const std::vector<std::string>& word_seg = q.word_seg;
	std::string cur_mention;
	cur_mention.reserve(q.question_sent.size());
	for (int i = 0; i < word_seg.size(); ++i){
		cur_mention.clear();
		for (int j = i; j < word_seg.size(); ++j){
			cur_mention += word_seg[j];
			cand_mention_node* node = generate_topic_mention(q, i, j + 1, cur_mention);
			if (node){
				_cand_mentions.push_back(node);
			}
		}
	}
	return 0;
}
 
std::vector<cand_entity_node*> CandGenerator::generate_entities_by_mention(
		const std::string& men_str, bool is_topic){
	std::vector<std::pair<int, std::string> > evec1 = 
			_kb_thread->get_entities_by_mention(men_str);
	std::vector<std::pair<int, std::string> > evec2;
	std::string no_mark_men = Utils::remove_book_mark(men_str);
	if (no_mark_men != ""){
		evec2 = _kb_thread->get_entities_by_mention(no_mark_men);
	}
	std::set<int> id_set;
	std::vector<cand_entity_node*> ret;
	for (int i = 0 ; i < evec1.size(); ++i){
		if (id_set.find(evec1[i].first) != id_set.end()){
			continue;
		}
		id_set.insert(evec1[i].first);
		cand_entity_node* node = new cand_entity_node();
		node->id = evec1[i].first;
		node->str = evec1[i].second;
		node->mention = men_str;
		node->is_topic = is_topic;
		if (is_topic){
			node->topic_mention = _tmp_father_mention;
		}
		ret.push_back(node);
		_cand_entities.push_back(node);
	}

	for (int i = 0 ; i < evec2.size(); ++i){
		if (id_set.find(evec2[i].first) != id_set.end()){
			continue;
		}
		id_set.insert(evec2[i].first);
		cand_entity_node* node = new cand_entity_node();
		node->id = evec2[i].first;
		node->str = evec2[i].second;
		node->mention = no_mark_men;
		node->is_topic = is_topic;
		if (is_topic){
			node->topic_mention = _tmp_father_mention;
		}
		ret.push_back(node);
		_cand_entities.push_back(node);
	}
	return ret;
}

cand_relation_node* CandGenerator::generate_cand_relation(const std::string& rel_str){
	auto iter = _cand_relations.find(rel_str);
	if (iter != _cand_relations.end()){
			return iter->second;
	}
	cand_relation_node* ret = new cand_relation_node();
	ret->str = rel_str;
	if (_preprocess_thread->wordseg(rel_str, ret->word_seg) != 0){
		Logger::logging("relation seg error: " + rel_str, "ERROR");
		delete ret;
		return NULL;
	}
	if (_preprocess_thread->word_seg2char_seg(ret->word_seg, ret->char_seg) != 0){
			Logger::logging("relation wordseg2charseg error: " + rel_str, "ERROR");
			delete ret;
			return NULL;
		}
	if (_preprocess_thread->char_seg2char_vec(ret->char_seg, ret->char_vec) != 0){
		Logger::logging("relation charseg2charvec error: " + rel_str, "ERROR");
		delete ret;
		return NULL;
	}
	_cand_relations[rel_str] = ret;
	return ret;	
}

cand_relation_chain_node* CandGenerator::generate_cand_relation_chain(cand_entity_node* entity,
		const std::pair<std::string, std::string>& po, cand_relation_chain_node* father){
	const std::string& rel_str = po.first;
	const std::string& tail_men = po.second;
	std::string rc_str;
	cand_mention_node* topic_mention = NULL;
	if (father == NULL){
		rc_str = rel_str;
		topic_mention = _tmp_father_mention;
	}else{
		rc_str = father->str + "->" + rel_str; 
		topic_mention = father->topic_mention;
	}
	
	cand_relation_chain_key key;
	key.men_st = topic_mention->word_st_index;
	key.men_en = topic_mention->word_en_index;
	key.str = rc_str;
	auto iter = _cand_relation_chains.find(key);
	if (iter != _cand_relation_chains.end()){
		return iter->second;
	}
	
	cand_relation_node* rel_node = generate_cand_relation(rel_str);
	if (!rel_node){
		Logger::logging("relation node create error!", "ERROR");
		return NULL;
	}
	
	cand_relation_chain_node* ret = new cand_relation_chain_node();
	ret->str = rc_str;
	ret->topic_mention = topic_mention;
	if (father){
		ret->relations = father->relations;
	}
	ret->relations.push_back(rel_node);
	_cand_relation_chains[key] = ret;
	return ret;
}


cand_answer_node* CandGenerator::generate_answer(cand_mention_node* mention, cand_entity_node* entity, 
		const std::pair<std::string, std::string>& po){
	cand_relation_chain_node* rc_node = generate_cand_relation_chain(entity, po, NULL);
	if (!rc_node){
		Logger::logging("cand_relation_chain_node create error!", "ERROR");
		return NULL;
	}
	cand_answer_node* ret = new cand_answer_node();
	ret->str = po.second;
	ret->topic_mention = mention;
	ret->topic_entity= entity;
	ret->relation_chain = rc_node;
	ret->mentions.push_back(mention->str);
	ret->mentions.push_back(po.second);
	ret->entities.push_back(entity);
	_cand_answers.push_back(ret);
	return ret;
}

cand_answer_node* CandGenerator::generate_answer(cand_entity_node* entity, 
		cand_answer_node* father, const std::pair<std::string, std::string>& po){
	cand_relation_chain_node* rc_node = generate_cand_relation_chain(entity, po, father->relation_chain);
	if (!rc_node){
		Logger::logging("cand_answer_node create error!", "ERROR");
		return NULL;
	}
	cand_answer_node* ret = new cand_answer_node();
	ret->str = po.second;
	ret->topic_mention = father->topic_mention;
	ret->topic_entity = father->topic_entity;
	ret->mentions = father->mentions;
	ret->entities = father->entities;
	ret->relation_chain = rc_node;
	ret->mentions.push_back(po.second);
	ret->entities.push_back(entity);
	_cand_answers.push_back(ret);
	return ret;
}

int CandGenerator::generate_cand_answers(){
	int ret = 0;
	int max_len = _s_limits["relation_chain_max_len"];
	for (int i = 0; i < _cand_mentions.size(); ++i){
		_tmp_father_mention = _cand_mentions[i];
		std::vector<cand_entity_node*> entities =generate_entities_by_mention(
				_cand_mentions[i]->str, true);
		for (int j = 0; j < entities.size(); ++j){
			const entity_kb_node* kb_entity = _kb_thread->get_entity_info_by_id(entities[j]->id);
			for (int k = 0; k < kb_entity->po_pairs.size(); ++k){
				cand_answer_node* node = generate_answer(_cand_mentions[i], 
						entities[j], kb_entity->po_pairs[k]);
				if (!node){
					Logger::logging("gen answers error!", "ERROR");
					ret = -1;
				}
			}
			_cand_topic_entities.push_back(entities[j]);
		}
	}
	
	int head = 0;
	while(head < _cand_answers.size()){
		cand_answer_node* cur_node = _cand_answers[head];
		if (cur_node->relation_chain->relations.size() >= max_len){
			++head;
			continue;
		}
		std::vector<cand_entity_node*> entities =generate_entities_by_mention(
				cur_node->str, false);
		for (int j = 0; j < entities.size(); ++j){
			const entity_kb_node* kb_entity = _kb_thread->get_entity_info_by_id(entities[j]->id);
			for (int k = 0; k < kb_entity->po_pairs.size(); ++k){
				cand_answer_node* node = generate_answer(entities[j], cur_node, kb_entity->po_pairs[k]);
				if (!node){
					Logger::logging("gen answers error!", "ERROR");
					ret = -1;
				}
			}
		}
		++head;
	}
	
	return ret;
}

int CandGenerator::generate(Question& q){
	clear();

	if (generate_cand_mentions(q) != 0){
		Logger::logging("cand mentions generate error: " + q.original_sent, "ERROR");
		return -1;
	}
	if (generate_cand_answers() != 0){
		Logger::logging("cand answers generate error: " + q.original_sent, "ERROR");
		return -1;
	}
	q.cands = new cand_info(_cand_mentions, _cand_topic_entities, _cand_entities, 
			_cand_relations, _cand_relation_chains, _cand_answers);
	return 0;
}
