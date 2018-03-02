#ifndef _WORD_CHAR_SEQ_DATA_FEED_H_
#define _WORD_CHAR_SEQ_DATA_FEED_H_
#include "data_feed.h"
#include "spp_hash_map.h"

struct word_char_seq_feed_dict{
	HashMap word2id;	
	HashMap char2id;
};

struct word_char_seq_input{
	std::vector<std::string> words;
	std::vector<std::vector<std::string> > chars;
};

class WordCharSeqDataFeed: public DataFeed{

public:
	virtual int thread_init(const std::map<std::string, std::string>& conf);
	virtual std::map<std::string, Tensor>& gen_input(void* org_data,
			void* dict, int index = 0);	
private:
	int _word_seq_len;
	int _char_seq_len;

public:
	static int global_feed_dict_init(const std::map<std::string, std::string>& conf,
			word_char_seq_feed_dict& dict);
	static void global_feed_dict_destroy(word_char_seq_feed_dict& dict);
};

#endif
