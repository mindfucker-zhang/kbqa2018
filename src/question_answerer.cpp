#include "question_answerer.h"
#include "utils.h"
#include "logger.h"

question_answerer_dict* QuestionAnswerer::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
		Logger::logging("QuestionAnswerer read conf map error!", "ERROR");
		return NULL;
	}
	
	question_answerer_dict* dict = new question_answerer_dict();
	char* keys[] = {"global_dict", "preprocessor", "knowledge_base", "cand_generator", 
				"entity_linker", "relation_chain_inferencer", "answer_reranker"};

	int key_num = 7;
	int ret = 0;
	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + "not in conf map!", "ERROR");
			ret = -1;
			break;
		}

		if (strcmp(keys[i], "global_dict") == 0){
			ret = GlobalDict::global_init(iter->second);
		
		}else if (strcmp(keys[i], "preprocessor") == 0){
			dict->pp_dict = Preprocessor::global_init(iter->second);
			if (!dict->pp_dict){
				ret = -1;
			}

		}else if (strcmp(keys[i], "knowledge_base") == 0){
			ret = KnowledgeBase::global_init(iter->second);
		
		}else if (strcmp(keys[i], "cand_generator") == 0){
			ret = CandGenerator::global_init(iter->second);
		
		}else if (strcmp(keys[i], "entity_linker") == 0){
			dict->el_dict = EntityLinker::global_init(iter->second);
			if (!dict->el_dict){
				ret = -1;
			}

		}else if (strcmp(keys[i], "relation_chain_inferencer") == 0){
			dict->rci_dict = RelationChainInferencer::global_init(iter->second);
			if (!dict->rci_dict){
				ret = -1;
			}

		}else if (strcmp(keys[i], "answer_reranker") == 0){
			dict->ar_dict = AnswerReranker::global_init(iter->second);
			if (!dict->ar_dict){
				ret = -1;
			}
		}

		if (ret != 0){
			Logger::logging(std::string(keys[i]) + "global init error!", "ERROR");	
			break;
		}
	}
	
	if (ret != 0){
		global_destroy(dict);
		return NULL;
	}
	return dict;
}

void QuestionAnswerer::global_destroy(question_answerer_dict*& dict){
	GlobalDict::global_destroy();
	Preprocessor::global_destroy(dict->pp_dict);
	KnowledgeBase::global_destroy();
	CandGenerator::global_destroy();
	EntityLinker::global_destroy(dict->el_dict);
	RelationChainInferencer::global_destroy(dict->rci_dict);
	AnswerReranker::global_destroy(dict->ar_dict);
	delete dict;
	dict = NULL;
}

int QuestionAnswerer::thread_init(question_answerer_dict* dict){
	if (_preprocessor.thread_init() != 0){
		Logger::logging("preprocessor thread init error!", "ERROR");
		return -1;
	}

	if (_knowledge_base.thread_init() != 0){
		Logger::logging("knowledge_base thread init error!", "ERROR");
		return -1;
	}

	if (_cand_generator.thread_init(&_preprocessor, &_knowledge_base) != 0){
		Logger::logging("cand_generator thread init error!", "ERROR");
		return -1;
	}

	if (_entity_linker.thread_init(dict->el_dict) != 0){
		Logger::logging("entity_linker thread init error!", "ERROR");
		return -1;
	}

	if (_relation_chain_inferencer.thread_init(dict->rci_dict) != 0){
		Logger::logging("relation_chain_inferencer thread init error!", "ERROR");
		return -1;
	}

	if (_answer_reranker.thread_init(dict->ar_dict) != 0){
		Logger::logging("answer_reranker thread init error!", "ERROR");
		return -1;
	}

	return 0;
}

void QuestionAnswerer::thread_destroy(){
	_preprocessor.thread_destroy();
	_knowledge_base.thread_destroy();
	_cand_generator.thread_destroy();
	_entity_linker.thread_destroy();
	_relation_chain_inferencer.thread_destroy();
	_answer_reranker.thread_destroy();
}

int QuestionAnswerer::question_answering(Question& q, question_answerer_dict* dict){
	if (_preprocessor.preprocess(q, dict->pp_dict) != 0){
		Logger::logging("preprocess error: " + q.original_sent, "ERROR");
		return -1;
	}

	if (_cand_generator.generate(q) != 0){
		Logger::logging("cand generate error: " + q.original_sent, "ERROR");
		return -1;
	}

	if (_entity_linker.process(q, dict->el_dict) != 0){
		Logger::logging("entity linking error: " + q.original_sent, "ERROR");
		return -1;
	}

	if (_relation_chain_inferencer.process(q, dict->rci_dict) != 0){
		Logger::logging("relation chaininference error: " + q.original_sent, "ERROR");
		return -1;
	}

	if (_answer_reranker.scoring(q, dict->ar_dict) != 0){
		Logger::logging("answer rerank error: " + q.original_sent, "ERROR");
		return -1;
	}

	return 0;
}
