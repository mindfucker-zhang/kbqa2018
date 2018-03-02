#ifndef _QUESTION_ANSWERER_H_
#define _QUESTION_ANSWERER_H_
#include "question.h"
#include "global_dict.h"
#include "preprocessor.h"
#include "knowledge_base.h"
#include "cand_generator.h"
#include "entity_linker.h"
#include "relation_chain_inferencer.h"
#include "answer_reranker.h"

struct question_answerer_dict{
	preprocess_dict* pp_dict;
	entity_linker_dict* el_dict;
	relation_chain_inferencer_dict* rci_dict;
	answer_reranker_dict* ar_dict;
};


class QuestionAnswerer{

public:
	int thread_init(question_answerer_dict* dict);
	void thread_destroy();
	int question_answering(Question& q, question_answerer_dict* dict);

private:
	Preprocessor _preprocessor;
	KnowledgeBase _knowledge_base;
	CandGenerator _cand_generator;
	EntityLinker _entity_linker;
	RelationChainInferencer _relation_chain_inferencer;
	AnswerReranker _answer_reranker;

public:
	static question_answerer_dict* global_init(const std::string& conf_path);
	static void global_destroy(question_answerer_dict*& dict);
};

#endif
