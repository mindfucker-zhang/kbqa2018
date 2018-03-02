#include "word_char_seq_data_feed.h"

int WordCharSeqDataFeed::global_feed_dict_init(const std::map<std::string, std::string>& conf,
		word_char_seq_feed_dict& dict){
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

	iter = conf.find("char2id");
	if (iter == conf.end()){
		Logger::logging("feed dict char2id path error: NULL!", "ERROR");
		return -1;
	}
	ret = HashMapFileStream::load_bin(iter->second, dict.char2id);
	if (ret != 0){
		Logger::logging("feed dict char2id path error:" + iter->second, "ERROR");
		return -1;

	}

	return 0;
}

void WordCharSeqDataFeed::global_feed_dict_destroy(word_char_seq_feed_dict& dict){
	dict.word2id.clear();
	dict.word2id.clear();
}

int WordCharSeqDataFeed::thread_init(const std::map<std::string, std::string>& conf){
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

	iter = conf.find("char_seq_len");
	if (iter == conf.end()){
		Logger::logging("data feed char_seq_len error!", "ERROR");
		return -1;
	}
	_char_seq_len = Utils::str2int(iter->second);
	if (!_char_seq_len){
		Logger::logging("data feed char_seq_len error!", "ERROR");
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

	shape.push_back(_char_seq_len);
	size *= _char_seq_len;
	data = new DataType[size * _batch_size];
	_memory_pool.push_back(data);
	for (int i = 0; i < _batch_size; ++i){
		_batch_input[i]["char_ids"] = Tensor(shape, false, data + size * i);
	}
	
	return 0;
}

std::map<std::string, Tensor>& WordCharSeqDataFeed::gen_input(void* org_data, void* dict, int index){
	std::vector<std::string>& words = ((word_char_seq_input*) org_data)->words;
	std::vector<std::vector<std::string> >& chars = ((word_char_seq_input*) org_data)->chars;
	HashMap& word2id = ((word_char_seq_feed_dict*) dict)->word2id;
	HashMap& char2id = ((word_char_seq_feed_dict*) dict)->char2id;
	Tensor& word_tensor = _batch_input[index]["word_ids"];
	Tensor& char_tensor = _batch_input[index]["char_ids"];
	DataType* word_data = word_tensor.get_var_data();
	DataType* char_data = char_tensor.get_var_data();
	for (int i = 0; i < _word_seq_len; ++i){
		if (i < words.size()){
			auto iter = word2id.find(words[i]);
			if (iter != word2id.end()){
				word_data[i] = iter->second;
			}else{
				word_data[i] = word2id["<unk>"];
			}

			std::vector<std::string>& cur_chars = chars[i];
			for (int j = 0; j < _char_seq_len; ++j){
				if (j < cur_chars.size()){
					auto iter = char2id.find(cur_chars[j]);
					if (iter != char2id.end()){
						char_data[i * _char_seq_len + j] = iter->second;
					}else{
						char_data[i * _char_seq_len + j] = char2id["<unk>"];
					}
				}else{
					char_data[i * _char_seq_len + j] = char2id["<null>"];
				}
			}
		}else{
			word_data[i] = word2id["<null>"];
			for (int j = 0; j < _char_seq_len; ++j){
				char_data[i * _char_seq_len + j] = char2id["<null>"];
			}
		}
	}
	
	return _batch_input[index];
}

