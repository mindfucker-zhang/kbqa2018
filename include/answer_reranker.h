#ifndef _ANSWER_RERANKER_H_
#define _ANSWER_RERANKER_H_

#include "ranker.h"
#include "question.h"

struct answer_reranker_dict{
	std::string ranker_type;
	void* ranker_dict;
};

class AnswerReranker{

public:
	int thread_init(answer_reranker_dict *dict);
	void thread_destroy();
	int scoring(Question& q, answer_reranker_dict* dict);

private:
	Ranker* _ranker;

public:
	static answer_reranker_dict* global_init(const std::string& conf_path);
	static void global_destroy(answer_reranker_dict*& dict);

};

#endif 
