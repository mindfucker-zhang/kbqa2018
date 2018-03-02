#ifndef _RELATION_CHAIN_SCORER_H_
#define _RELATION_CHAIN_SCORER_H_
#include "question.h"
#include "semantics_embedding_net.h"
#include "word_char_seq_data_feed.h"

class RelationChainScorer{
	
public:
	virtual int thread_init(void* dict) = 0;
	virtual void thread_destroy() = 0;
	virtual void clear() = 0;
	virtual int scoring(Question &q, void* dict, const std::string& key) = 0;
	virtual ~RelationChainScorer(){}
};

struct relation_cnn_scorer_dict{
	net_dict* sent_cnn_net_dict;
	net_dict* rc_cnn_net_dict;
	std::map<std::string, std::string> sent_feed_conf;
	std::map<std::string, std::string> rc_feed_conf;
	word_char_seq_feed_dict feed_dict;
};

class RelationChainCNNScorer: public RelationChainScorer{

public:
	virtual int thread_init(void* dict);
	virtual void thread_destroy();
	virtual void clear();
	virtual int scoring(Question &q, void* dict, const std::string& key);

private:
	void gen_seq_input(cand_relation_chain_node* cand, word_char_seq_input& seq_input);

private:
	SemanticsEmbeddingNet _sent_net;
	SemanticsEmbeddingNet _rc_net;
	WordCharSeqDataFeed _sent_data_feed;
	WordCharSeqDataFeed _rc_data_feed;
	std::vector<word_char_seq_input> _sent_inputs;
	std::vector<word_char_seq_input> _rc_inputs;
	int _sent_batch_size;
	int _rc_batch_size;

public:
	static relation_cnn_scorer_dict* global_init(const std::string& conf_path);
	static void global_destroy(void*& dict);
};

#endif
