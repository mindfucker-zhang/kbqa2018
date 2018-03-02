#include <iostream>
#include <exception>
#include <map>
#include <string.h>
#include "preprocessor.h"
#include "word_seg.h"
#include "utils.h"
#include "logger.h"

preprocess_dict* Preprocessor::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	int ret = Utils::read_conf_map(conf_path, conf_map);
	if (ret != 0){
		Logger::logging("preprocee conf path error: " + conf_path, "ERROR");
		return NULL;
	}
	preprocess_dict* dict = new preprocess_dict();
	char* keys[] = {"prefix", "suffix", "wordseg"};
	int key_num = 3;
	std::string err_key;

	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + 
					" not in preprocess conf_map","ERROR");
			ret = -1;
			break;
		}
		if (strcmp(keys[i], "prefix") == 0){
			ret = Utils::load_dict(iter->second, dict->prefix);
			dict->max_prefix_len = 0;
			auto siter = dict->prefix.begin();
			for (; siter != dict->prefix.end(); ++siter){
				dict->max_prefix_len = (siter->size() > dict->max_prefix_len)? 
						siter->size(): dict->max_prefix_len;
			}

		}else if (strcmp(keys[i], "suffix") == 0){
			ret = Utils::load_dict(iter->second, dict->suffix);
			dict->max_suffix_len = 0;
			auto siter = dict->suffix.begin();
			for (; siter != dict->suffix.end(); ++siter){
				dict->max_suffix_len = (siter->size() > dict->max_suffix_len)?
						siter->size(): dict->max_suffix_len;
			}

		}else if (strcmp(keys[i], "wordseg") == 0){
			ret = WordSeg::global_init(iter->second);
		}

		if (ret != 0){
			err_key = keys[i];
			break;
		}
	}

	if (err_key != ""){
		Logger::logging(err_key + "global init error: " + conf_map[err_key], "ERROR");
	}
	if (ret != 0){
		Logger::logging("preprocess global init error!", "ERROR");
		global_destroy(dict);
	}
	return dict;
}

void Preprocessor::global_destroy(preprocess_dict*& dict){
	WordSeg::global_destroy();
	delete dict;
	dict = NULL;
}

int Preprocessor::thread_init(){
	int ret = _word_seg_thread.thread_init();
	if (ret != 0){
		Logger::logging("preprocessor thread init error!", "ERROR");
	}
	return ret;
}

void Preprocessor::thread_destroy(){
	_word_seg_thread.thread_destroy();
	return;
}

void Preprocessor::to_lower(const std::string &src_sent, std::string &tgt_sent){
	tgt_sent.clear();
	int diff = 'a' - 'A';
	size_t i = 0;
	while(i < src_sent.size()){
		char ch = src_sent[i];
		if ((ch & 0xff)>= 0x00 && (ch & 0xff) <= 0x7F){
			if (ch >= 'A' && ch <= 'Z'){
				tgt_sent.push_back(ch + diff);	
			}else{
				tgt_sent.push_back(ch);
			}
			i += 1;

		}else if ((src_sent[i + 1] & 0xff) >= 0x40){
			tgt_sent.push_back(src_sent[i]);
			tgt_sent.push_back(src_sent[i + 1]);
			i += 2;

		}else{
			tgt_sent.push_back(src_sent[i]);
			tgt_sent.push_back(src_sent[i + 1]);
			tgt_sent.push_back(src_sent[i + 2]);
			tgt_sent.push_back(src_sent[i + 3]);
			i += 4;
		}
	}
}

void Preprocessor::remove_prefix(const std::string &src_sent, std::string &tgt_sent,
			const std::set<std::string>& prefix, int max_prefix_len){
	tgt_sent.clear();
	size_t cur_len = 0;
	size_t max_len = (max_prefix_len < src_sent.size())? 
			max_prefix_len: src_sent.size();
	for (size_t i = 1; i <= max_len; ++i){
		if (prefix.find(src_sent.substr(0, i)) != prefix.end()){
			cur_len = i;
		}
	}
	tgt_sent = src_sent.substr(cur_len, src_sent.size() - cur_len);
}

void Preprocessor::remove_suffix(const std::string &src_sent, std::string &tgt_sent,
		const std::set<std::string>& suffix, int max_suffix_len){
	tgt_sent.clear();
	size_t cur_len = 0;
	size_t max_len = (max_suffix_len < src_sent.size())?
			max_suffix_len : src_sent.size();
	for (size_t i = 1; i <= max_len; ++i){
		if (suffix.find(src_sent.substr(src_sent.size() - i, i)) != suffix.end()){
			cur_len = i;	
		}
	}
	tgt_sent = src_sent.substr(0, src_sent.size() - cur_len);
}

int Preprocessor::wordseg(const std::string& sent, std::vector<std::string>& word_seg){
	int ret = _word_seg_thread.seg_sentence(sent, word_seg);
	if (ret != 0){
		Logger::logging("wordseg error: " + sent, "ERROR");
	}
	return ret;
}

int Preprocessor::word_seg2char_seg(const std::vector<std::string>& word_seg,
		std::vector<std::vector<std::string> >& char_seg){
	char_seg.resize(word_seg.size());
	for (int i = 0; i < word_seg.size(); ++i){
		if (Utils::word2char_gbk(word_seg[i], char_seg[i]) != 0){
			return -1;
		}
	}
	return 0;
}

int Preprocessor::char_seg2char_vec(const std::vector<std::vector<std::string> >& char_seg,
                        std::vector<std::string>& char_vec){
	Utils::merge_vector(char_seg, char_vec);
	return 0;
}

int Preprocessor::preprocess(Question& question, preprocess_dict* dict){
	to_lower(question.original_sent, question.lower_sent);
	std::string tmp;
	remove_prefix(question.lower_sent, tmp, dict->prefix, dict->max_prefix_len);
	remove_suffix(tmp, question.question_sent, dict->suffix, dict->max_suffix_len);	
	int ret = wordseg(question.question_sent, question.word_seg) == 0 && 
			word_seg2char_seg(question.word_seg, question.char_seg);
	if (ret != 0){
		Logger::logging("preprocess error: " + question.question_sent, "ERROR");
		return -1;
	}
	char_seg2char_vec(question.char_seg, question.char_vec);
	return 0;
}

