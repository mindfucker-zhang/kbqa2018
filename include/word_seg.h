#ifndef _WORD_SEG_H_
#define _WORD_SEG_H_
#include <string>
#include <vector>
#include "NLPIR.h"
#define SEG_SENT_MAX_LEN 10240

class WordSeg{
public:
	WordSeg();
	int thread_init();
	void thread_destroy();
	int seg_sentence(const std::string& sent, std::vector<std::string>& word_seg);
	~WordSeg();
		
private:
	CNLPIR* _thread_ptr;
	char _sSentence[SEG_SENT_MAX_LEN];

public:
	static int global_init(const std::string& dict_path);
	static void global_destroy();	

};


#endif
