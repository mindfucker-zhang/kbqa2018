#include <iostream>
#include "utils.h"
#include <pthread.h>
#include "logger.h"
#include "preprocessor.h"

preprocess_dict* pdict; 
pthread_mutex_t global_mutex;
int qid;

void* run_preprocess_thread(void* dict){
	Question question;
	Preprocessor thread;
	if (thread.thread_init() != 0){
		Logger::logging("thread init error", "ERROR");
		return NULL;
	}
	
	std::string line;
	while(1){
		pthread_mutex_lock(&global_mutex);
                if (std::cin.eof() || !std::getline(std::cin, line)){
                        pthread_mutex_unlock(&global_mutex);
                        break;
                }
		question.qid = qid++;
                pthread_mutex_unlock(&global_mutex);
		question.original_sent = line;
		
		int ret = thread.preprocess(question, pdict);
		if (ret != 0){
			Logger::logging("preprocessing error!", "ERROR");
			break;
		}
		std::string ostr;
		std::string tmp_str;
		ostr = question.original_sent + "\n" + question.lower_sent + "\n" + 
				question.question_sent + "\n";
		Utils::join(question.word_seg, " ||| ", tmp_str);
		ostr += tmp_str + "\n";
		Utils::join(question.char_vec, " ||| ", tmp_str);
		ostr += tmp_str + "\n";
		for (int i = 0; i < question.char_seg.size(); ++i){
			Utils::join(question.char_seg[i], " | ",  tmp_str);
			ostr += tmp_str;
			if (i == question.char_seg.size() - 1){
				ostr += "\n";
			}else{
				ostr += " ||| ";
			}
		}
		pthread_mutex_lock(&global_mutex);
		std::cout << "qid: " << question.qid << "\n" << ostr << "\n";
		pthread_mutex_unlock(&global_mutex);
	}
	thread.thread_destroy();
}

int main(int argc, char *argv[]){
	pthread_mutex_init(&global_mutex, NULL);
	pthread_t pids[20];
	int thread_num = 1;
	
	if (argc < 2){
		Logger::logging("arguement error !", "ERROR");
		return -1;
	}
	if (argc >= 3){
		thread_num = Utils::str2int(argv[2]);
	}
	
	pdict = Preprocessor::global_init(argv[1]);
	if (pdict == NULL){
		Logger::logging("global init error !", "ERROR");
		return -1;
	}

	for (int i = 0; i < thread_num; ++i){
		if (pthread_create(&pids[i], 0, run_preprocess_thread, NULL) != 0){
			Logger::logging("pthread init error", "ERROR");
			return -1;
		}
	}
	for (int i = 0; i < thread_num; ++i){
		pthread_join(pids[i], NULL);
	}

	Preprocessor::global_destroy(pdict);
	return 0;
}

