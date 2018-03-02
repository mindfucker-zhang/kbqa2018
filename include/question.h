#ifndef _QUESTION_H_
#define _QUESTION_H_

#include <string>
#include <vector>
#include <map>
#include "cand_node.h"

struct  Question{
	int qid;
	std::string original_sent;
	std::string lower_sent;
	std::string removed_sent;
	std::string question_sent;
	std::vector<std::string> word_seg;
	std::vector<std::vector<std::string> > char_seg;
	std::vector<std::string> char_vec;
	cand_info* cands;

	void clear(){
		original_sent.clear();
		lower_sent.clear();
		removed_sent.clear();
		question_sent.clear();
		word_seg.clear();
		char_seg.clear();
		char_vec.clear();
		if (cands){
			delete cands;
			cands = NULL;
		}
	}
	
};

#endif
