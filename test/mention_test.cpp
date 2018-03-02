#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <time.h>
#include "mention_rank_net.h"
#include "word_seq_data_feed.h"
#include "logger.h"
#include "utils.h"

int gen_bin_hash_map(const std::string& fpath, const std::string& tpath){
	HashMap hm;
	int ret = HashMapFileStream::load(fpath, hm, false, 12000);
	if (ret != 0){
		Logger::logging("hash read error", "ERROR");
		return -1;
	}
	ret = HashMapFileStream::save_bin(tpath, hm);
	if (ret != 0){
		Logger::logging("hash save error", "ERROR");
		return -1;
	}
	return 0;
}

int gen_bin_net(){
	net_model model;
	clock_t t = clock();
	int ret = Net::load_model("mention_rank_model", model);
	std::cout << (clock() - t) * 1.0 / CLOCKS_PER_SEC << "\n";
	if (ret){
		Logger::logging("load net error", "ERROR");
		return -1;
	}
	ret = Net::save_bin_model("mention_rank_bin", model);
	if (ret){
		Logger::logging("load save error", "ERROR");
		return -1;
	}
		
	auto params = model.params;
	for (auto iter = params.begin(); iter != params.end(); ++iter){
		for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2){
			std::cout << iter->first << "/" << iter2->first << ": " << 
				iter2->second.get_size() << "\n";
			const DataType* data = iter2->second.get_data();
			for (int i = 0; i < iter2->second.get_size(); ++i){
				std::cout << data[i] << " ";
				if (i % 6 == 5){
					std::cout << "\n";
				}
			}
			std::cout << "\n";
		}
	}
	
	return 0;
}

void load_model(){
	net_model model;
	clock_t t = clock();
	int ret = Net::load_bin_model("mention_rank_bin", model);
	if (ret){
		Logger::logging("load bin error", "ERROR");
	}
	std::cout << (clock() - t) *1.0 / CLOCKS_PER_SEC << "\n";
	auto params = model.params;
        for (auto iter = params.begin(); iter != params.end(); ++iter){
                for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2){
                        std::cout << iter->first << "/" << iter2->first << ": " <<
                                iter2->second.get_size() << "\n";
                        const DataType* data = iter2->second.get_data();
                        for (int i = 0; i < iter2->second.get_size(); ++i){
                                std::cout << data[i] << " ";
                                if (i % 6 == 5){
                                        std::cout << "\n";
                                }
                        }
                        std::cout << "\n";
                }
        }
} 

void debug_test(){
	DataType* data = new DataType[3];
	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	std::vector<int> shape;
	shape.push_back(3);
	Tensor t(shape, false, data);
	std::ofstream fo("test.bin");
	boost::archive::binary_oarchive bin_fo(fo);
	bin_fo << t;
	fo.close();
	//delete data;
	//data = NULL;
	std::ifstream fin("test.bin");
	boost::archive::binary_iarchive bin_fin(fin);	
	Tensor t2;
	bin_fin >>t2;
	const DataType* d = t2.get_data();
	std::cout << d[0] << d[1] << d[2] << "\n";	
}

void mention_rank_test();

int main(){
	Logger::global_init();
	//gen_bin_net();
	//load_model();
	//debug_test();
	mention_rank_test();
}

void mention_rank_test(){
	int batch_size = 20;
	net_model model;
	std::map<std::string, std::map<std::string, std::vector<int> > > para_conf;
        int ret = Net::load_bin_model("mention_rank_bin", model);
	ret = Net::load_para_conf("para_conf", para_conf);
	Net* mr_net = new MentionRankNet();
	mr_net->thread_init(para_conf, batch_size);

	std::map<std::string, std::string> conf;
	conf["word_seq_len"] = "20";
	conf["batch_size"] = "20";
	conf["word2id"] = "word2id.bin";
	
	word_seq_feed_dict feed_dict;	
	WordSeqDataFeed::global_feed_dict_init(conf, feed_dict);
	DataFeed* feed = new WordSeqDataFeed();
	feed->thread_init(conf);
	
	std::vector<word_seq_input> inputs;
	inputs.resize(batch_size);
	std::vector<void*> void_inputs;
	for (int i = 0; i < batch_size; ++i){
		void_inputs.push_back((void*)&(inputs[i]));
	}
	std::vector<int> sts;
	sts.resize(batch_size);
	std::vector<int> ens;
	ens.resize(batch_size);
	int num = 0;
	std::string line;

	time_t t1, t2;
	time(&t1);

	while(!std::cin.eof() && std::getline(std::cin, line)){
		int index = num % batch_size;
		++num;
		std::vector<std::string> elems;
		Utils::split(line, "\t", elems);
		Utils::split(elems[0], " ||| ", inputs[index].words);
		sts[index] = Utils::str2int(elems[1]);
		ens[index] = Utils::str2int(elems[2]);
		if (index == batch_size -1){
			std::vector<std::map<std::string, Tensor> >& input_tensor = 
				feed->gen_batch_input(void_inputs, (void*) &feed_dict);
			const std::vector<std::map<std::string, Tensor> >& out_tensor =
				 mr_net->run_batch(input_tensor, model.params);
			
			for (int i = 0; i < batch_size; ++i){
				const DataType* res = out_tensor[i].find("probs")->second.get_data();
				DataType score = res[sts[i] * 3 + 0] * res[ens[i] * 3 + 1];
        			std::cout << score << "\n";
			}
		}
	}
	num = num % batch_size;
	if (num){
		std::vector<void*> voids;
		for (int i = 0; i < num; ++i){
			voids.push_back((void*)&(inputs[i]));
		}
		std::vector<std::map<std::string, Tensor> >& input_tensor =
                		feed->gen_batch_input(voids, (void*) &feed_dict);
                const std::vector<std::map<std::string, Tensor> >& out_tensor =
                		mr_net->run_batch(input_tensor, model.params);
		for (int i = 0; i < num; ++i){
			const DataType* res = out_tensor[i].find("probs")->second.get_data();
			DataType score = res[sts[i] * 3 + 0] * res[ens[i] * 3 + 1];
			std::cout << score << "\n";
		}
	}

	time(&t2);
	std::cerr << difftime(t2, t1) << "\n";
}	
	/*const DataType* emb = out_tensor.find("emb")->second.get_data();
	const DataType* lstm = out_tensor.find("lstm")->second.get_data();

	std::cout << "fstm:\n";
	for (int i = 0; i < 20; ++i){
                for (int j = 0; j < 128; ++j){
			std::cout << lstm[i*256+j] << " ";
                }
                std::cout << "\n";
	}

	std::cout << "bstm:\n";
        for (int i = 0; i < 20; ++i){
                for (int j = 128; j < 256; ++j){
                        std::cout << lstm[i*256+j] << " ";
                }
                std::cout << "\n";
        }


	std::cout << "emb:\n";
	for (int i = 0; i < 20; ++i){
		for (int j = 0; j < 256; ++j){
			std::cout << emb[i*256+j] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "probs:\n";
	for (int i = 0; i < 20; ++i){
		for (int j = 0; j < 3; ++j){
			std::cout << res[i*3+j] << " ";
		}
		std::cout << "\n";
	}*/
