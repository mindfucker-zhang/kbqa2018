#ifndef _MENTION_SCORER_H_
#define _MENTION_SCORER_H_
#include "question.h"
#include "mention_rank_net.h"
#include "word_seq_data_feed.h"

class MentionScorer{
	
public:
	virtual int thread_init(void* dict) = 0;
	virtual void thread_destroy() = 0;
	virtual void clear() = 0;
	virtual int scoring(Question &q, void* dict, const std::string& key) = 0;
	virtual ~MentionScorer(){}
};

struct mention_net_scorer_dict{
	net_dict* mention_net_dict;
	std::map<std::string, std::string> feed_conf;
	word_seq_feed_dict feed_dict;
};

class MentionNetScorer: public MentionScorer{

public:
	virtual int thread_init(void* dict);
	virtual void thread_destroy();
	virtual void clear();
	virtual int scoring(Question &q, void* dict, const std::string& key);


private:
	MentionRankNet _net;
	WordSeqDataFeed _data_feed;
	word_seq_input _input;
	
public:
	static mention_net_scorer_dict* global_init(const std::string& conf_path);
	static void global_destroy(void*& dict);
};

#endif