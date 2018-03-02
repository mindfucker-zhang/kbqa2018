#ifndef _PREPROCESSOR_H_
#define _PREPROCESSOR_H_
#include <set>
#include <string>
#include "word_seg.h"
#include "question.h"

struct preprocess_dict{
	std::set<std::string> prefix;
        std::set<std::string> suffix;
        size_t max_prefix_len;
        size_t max_suffix_len;	
};


class Preprocessor{

public:
	int thread_init();
	void thread_destroy();
	int preprocess(Question& question, preprocess_dict* dict);
	void to_lower(const std::string &src_sent, std::string &tgt_sent); 
        void remove_prefix(const std::string &src_sent, std::string &tgt_sent,
			const std::set<std::string>& prefix, int max_prefix_len);
        void remove_suffix(const std::string &src_sent, std::string &tgt_sent,
			const std::set<std::string>& suffix, int max_suffix_len);
	int wordseg(const std::string& sent, std::vector<std::string>& word_seg);
	int word_seg2char_seg(const std::vector<std::string>& word_seg,
			std::vector<std::vector<std::string> >& char_seg);
	int char_seg2char_vec(const std::vector<std::vector<std::string> >& char_seg,
			std::vector<std::string>& char_vec);

public:
	static preprocess_dict* global_init(const std::string& conf_path);
	static void global_destroy(preprocess_dict*& dict);

private:
	WordSeg _word_seg_thread;

};

#endif
