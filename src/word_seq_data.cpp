#include "word_seq_data_feed.h"
#include "logger.h"
#include "utils.h"

int WordSeqDataFeed::global_feed_dict_init(const std::map<std::string, std::string>& conf,
		word_seq_feed_dict& dict){
	auto iter = conf.find("word2id");
	if (iter == conf.end()){
		Logger::logging("feed dict word2id path error: NULL!", "ERROR");
		return -1;
	}
	int ret = HashMapFileStream::load_bin(iter->second, dict.word2id);
	if (ret != 0){
		Logger::logging("feed dict word2id path error:" + iter->second, "ERROR");
		return -1;
	}
	return 0;
}

void WordSeqDataFeed::global_feed_dict_destroy(word_seq_feed_dict& dict){
	dict.word2id.clear();
}

int WordSeqDataFeed::thread_init(const std::map<std::string, std::string>& conf){
	int ret = DataFeed::thread_init(conf);
	if (ret != 0){
		return -1;
	}
	auto iter = conf.find("word_seq_len");
	if (iter == conf.end()){
		Logger::logging("data feed word_seq_len error!", "ERROR");
		return -1;
	}
	_word_seq_len = Utils::str2int(iter->second);	
	if (!_word_seq_len){
		Logger::logging("data feed word_seq_len error!", "ERROR");
		return -1;
	}

	std::vector<int> shape;
	shape.push_back(_word_seq_len);
	int size = _word_seq_len;
	DataType* data = new DataType[size * _batch_size];
	_memory_pool.push_back(data);
	for (int i = 0; i < _batch_size; ++i){
		_batch_input[i]["word_ids"] = Tensor(shape, false, data + size * i);
	}
	return 0;
}

std::map<std::string, Tensor>& WordSeqDataFeed::gen_input(void* org_data, void* dict, int index){
	std::vector<std::string>& words = ((word_seq_input*) org_data)->words;
	HashMap& word2id = ((word_seq_feed_dict*) dict)->word2id;
	Tensor& tensor = _batch_input[index]["word_ids"];
	DataType* data = tensor.get_var_data();
	data[0] = word2id["<begin>"];
	for (int i = 1; i < _word_seq_len; ++i){
		int index = i - 1;
		if (index < words.size()){
			auto iter = word2id.find(words[index]);
			if (iter != word2id.end()){
				data[i] = iter->second;
			}else{
				data[i] = word2id["<unk>"];
			}
			
		}else if (index == words.size()){
			data[i] = word2id["<end>"];
		}
		else{
			data[i] = word2id["<null>"];
		}
	}
	return _batch_input[index];
}


