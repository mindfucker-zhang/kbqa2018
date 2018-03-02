#ifndef _ENTITY_SCORER_H_
#define _ENTITY_SCORER_H_
#include "question.h"

class EntityScorer{
	
public:
	virtual int thread_init(void* dict) = 0;
	virtual void thread_destroy() = 0;
	virtual void clear() = 0;
	virtual int scoring(Question &q, void* dict, const std::string& key) = 0;
};



#endif