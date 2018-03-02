#include <iostream>
#include <sstream>
#include <pthread.h>
#include "logger.h"
#include "utils.h"
#include "time.h"
#include "question_answerer.h"

question_answerer_dict* qa_dict;
pthread_mutex_t global_mutex;
int qid;

void* run_qa_thread(void* dict){
	Question question;
	QuestionAnswerer thread;
	if (thread.thread_init(qa_dict) != 0){
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
		std::cerr << "qid: " << qid - 1 << std::endl;
		pthread_mutex_unlock(&global_mutex);
		question.clear();
		question.original_sent = line;
		
		int ret = thread.question_answering(question, qa_dict);
		if (ret != 0){
			Logger::logging("question_answering error!", "ERROR");
			break;
		}
		
		std::stringstream ss;
		ss << "question: " << question.original_sent << "\n";
		/*std::vector<cand_answer_node*>&  cands = question.cands->cand_answers;
		for (int i = 0; i < cands.size(); ++i){
			ss << cands[i]->topic_mention->str << "\t" << cands[i]->topic_entity->str << "\t"
					<<  cands[i]->str << "\n";
			std::string tmp;
			Utils::join(cands[i]->mentions, "->", tmp);
			ss << "mentions: " << tmp << "\n";
			ss << "entities: ";
			for (int j = 0; j < cands[i]->entities.size(); ++j){
				ss << cands[i]->entities[j]->str << "->";
			}
			ss << "\n";
			ss << "relations: " << cands[i]->relation_chain->str << "\n";		
 
		}*/

		std::vector<cand_mention_node*>&  cands = question.cands->cand_mentions;
		for (int i = 0; i < cands.size(); ++i){
			ss << "mention: " << cands[i]->str; 
			ss << "pat: " << cands[i]->pat << "\n";
			std::string tmp;
			Utils::join(cands[i]->pat_word_seg, " | ", tmp);
			ss << "word seg: " << tmp << "\n";
			ss << "pat_word_index: " << cands[i]->pat_word_index << "\n";
			Utils::join(cands[i]->pat_char_vec, " | ", tmp);
			ss << "char_vec: "<<  tmp << "\n";
			ss << "pat_char_index: " << cands[i]->pat_char_index << "\n";
			ss << "pat_char_seg: ";
			for (int j = 0; j < cands[i]->pat_char_seg.size(); ++j){
				Utils::join(cands[i]->pat_char_seg[j], " | ", tmp);
				ss << tmp;
				if (j != cands[i]->pat_char_seg.size() - 1){
					ss << "  ";
				}else{
					ss << "\n";
				}
			}
		}
		
		ss << "\n";
		pthread_mutex_lock(&global_mutex);
		std::cout << ss.str() << std::endl;
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
	
	time_t t1, t2;
	time(&t1);
	qa_dict = QuestionAnswerer::global_init(argv[1]);
	if (qa_dict == NULL){
		Logger::logging("global init error !", "ERROR");
		return -1;
	}
	Logger::logging("global init done !", "Notice");
	time(&t2);

	std::cerr << "load time: " << difftime(t2, t1) << std::endl;
	for (int i = 0; i < thread_num; ++i){
		if (pthread_create(&pids[i], 0, run_qa_thread, NULL) != 0){
			Logger::logging("pthread init error", "ERROR");
			return -1;
		}
	}
	for (int i = 0; i < thread_num; ++i){
		pthread_join(pids[i], NULL);
	}

	QuestionAnswerer::global_destroy(qa_dict);
	return 0;
}

