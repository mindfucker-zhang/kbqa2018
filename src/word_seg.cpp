#include <string.h>
#include <iostream>
#include "word_seg.h"
#include "logger.h"

WordSeg::WordSeg(){
	_thread_ptr = NULL;
}

int WordSeg::global_init(const std::string& dict_path){
	if(!NLPIR_Init(dict_path.c_str())){
		return -1;	
	}
	return 0;	
}

void WordSeg::global_destroy(){
	NLPIR_Exit();
}

int WordSeg::thread_init(){
	_thread_ptr = NULL;
	_thread_ptr = new CNLPIR();
	if (_thread_ptr == NULL){
		Logger::logging("wordseg thread init error", "ERROR");
		return -1;
	}
	return 0;
}

void WordSeg::thread_destroy(){
	if (_thread_ptr){
		delete _thread_ptr;
		_thread_ptr = NULL;
	}
}

WordSeg::~WordSeg(){
	thread_destroy();	
}

int WordSeg::seg_sentence(const std::string& sent, std::vector<std::string>& word_seg){
	int word_num = 0;
	const result_t* seg_out = NULL;
	strcpy(_sSentence, sent.c_str());
	seg_out = NLPIR_ParagraphProcessA(_sSentence, &word_num);
	if (seg_out == NULL){
		Logger::logging("word seg error: " + sent, "ERROR");
		return -1;
	}
	word_seg.resize(word_num);
        for (int i = 0; i < word_num; ++i){
		word_seg[i] = sent.substr(seg_out[i].start, seg_out[i].length);
	}
	return 0;
}
