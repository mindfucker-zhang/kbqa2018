#ifndef _RANKER_H_
#define _RANKER_H_
#include <map>
#include <string>

class Ranker{
	
public:
	virtual int thread_init(void* dict = NULL) = 0;
	virtual void thread_destroy() = 0;
	virtual void clear() = 0;
	virtual int scoring(const std::map<std::string, float>& feats, float& score, void* dict) = 0;
	virtual ~Ranker(){}
};

struct liner_ranker_dict{
	std::map<std::string, float> weights;
};

class LinerRanker: public Ranker{

public:
	virtual int thread_init(void* dict = NULL);
	virtual void thread_destroy();
	virtual void clear();
	virtual int scoring(const std::map<std::string, float>& feats, float& score, void* dict);

public:
	static liner_ranker_dict* global_init(const std::string& path);
	static void global_destroy(void*& dict);
};


#endif

