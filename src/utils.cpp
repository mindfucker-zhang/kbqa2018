#include <sstream>
#include <fstream>
#include "logger.h"
#include "utils.h"
#include "global_dict.h"

void Utils::split(const std::string& str, const std::string& tag, 
		std::vector<std::string>& res){
	res.clear();
	size_t st = 0;
	while(st < str.size()){
		size_t pos = str.find(tag, st);
		if (pos == std::string::npos){
			break;
		}
		res.push_back(str.substr(st, pos - st));
		st = pos + tag.size();
	}
	if (st < str.size()){
		res.push_back(str.substr(st, str.size() - st));
	}
}

std::vector<std::string> Utils::split(const std::string& str, const std::string& tag){
	std::vector<std::string> res;
	size_t st = 0;
	while(st < str.size()){
		size_t pos = str.find(tag, st);
		if (pos == std::string::npos){
			break;
		}
		res.push_back(str.substr(st, pos - st));
		st = pos + tag.size();
	}
	if (st < str.size()){
		res.push_back(str.substr(st, str.size() - st));
	}
	return res;
}

int Utils::str2int(const std::string& str){
	std::stringstream ss;
	ss << str;
	int ret;
	ss >> ret;
	return ret;
}

float Utils::str2float(const std::string& str){
	std::stringstream ss;
	ss << str;
	float ret;
	ss >> ret;
	return ret;
}

bool Utils::is_empty(const std::string& str){
	for (int i = 0; i < str.size(); ++i){
		if (str[i] != ' ' && str[i] != '\n' && str[i] != '\t' && str[i] != '\r'){
			return false;
		}
	}
	return true;
}

int Utils::read_conf_map(const std::string& conf_file,
  		std::map<std::string, std::string>& conf_map){
        conf_map.clear();
        std::ifstream fin(conf_file.c_str());
        if (!fin.is_open()){
                return -1;
        }
        std::string line;
        std::vector<std::string> res_pair;
        while(!fin.eof() && std::getline(fin, line)){
                if (line.size() == 0 || line[0] == '#'){
                        continue;
                }
                split(line, ": ", res_pair);
                if (res_pair.size() < 2){
                        continue;
                }
                conf_map[res_pair[0]] = res_pair[1];
        }
        return 0;
}

int Utils::load_dict(const std::string& path, std::set<std::string>& res){
	std::ifstream fin(path.c_str());
	if (!fin.is_open()){
		return -1;
	}
	res.clear();
	std::string line;
	while(!fin.eof() && std::getline(fin, line)){
		if (!line.size()){
			continue;
		}
		res.insert(line);
	}
	return 0;
}

int Utils::load_dict(const std::string& path, std::map<std::string, int>& res,
		const std::string& sp){
	std::ifstream fin(path.c_str());
        if (!fin.is_open()){
                return -1;
        }
        res.clear();
        std::string line;
	std::vector<std::string> kv;
        while(!fin.eof() && std::getline(fin, line)){
                if (!line.size()){
                        continue;
                }
		split(line, sp, kv);
		if (kv.size() < 2){
			continue;
		}
		res[kv[0]] = Utils::str2int(kv[1]);
	}
	return 0;	
}

int Utils::load_dict(const std::string& path, std::map<std::string, float>& res,
		const std::string& sp){
	std::ifstream fin(path.c_str());
	if (!fin.is_open()){
		return -1;
	}
	res.clear();
	std::string line;
	std::vector<std::string> kv;
	while(!fin.eof() && std::getline(fin, line)){
		if (!line.size()){
			continue;
		}
		split(line, sp, kv);
		if (kv.size() < 2){
			continue;
		}
		res[kv[0]] = Utils::str2float(kv[1]);
	}
	return 0;	
}

int Utils::word2char_gbk(const std::string& word, std::vector<std::string>& char_vec){
	char_vec.clear();
        int pos = 0;
        while(pos < word.size()){
                if ((word[pos] & 0xff)>= 0x00 && (word[pos] & 0xff) <= 0x7F){
                        char_vec.push_back(word.substr(pos, 1));
                        pos += 1;
                }else if (pos + 2 > word.size()){
                        Logger::logging("gbk word error!", "ERROR");
                        return -1;
                }else if ((word[pos + 1] & 0xff) >= 0x40){
                        char_vec.push_back(word.substr(pos, 2));
                        pos += 2;
                }else if (pos + 4 > word.size()){
			Logger::logging("gbk word error!", "ERROR");
                        return -1;
                }else{
                        char_vec.push_back(word.substr(pos, 4));
                        pos += 4;
                }
        }
	return 0;
}


void Utils::merge_vector(const std::vector<std::vector<std::string> >& vec,
		std::vector<std::string>& res){
	int size = 0;
	for (int i = 0; i < vec.size(); ++i){
		size += vec[i].size();
	}
	res.resize(size);
	int index = 0;
	for (int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			res[index++] = vec[i][j];
		}
	}
}

void Utils::join(const std::vector<std::string> &str_vec, const std::string &pt,
		std::string &res_str){
        res_str.clear();
        if (str_vec.size() == 0){
                return;
        }
        for (size_t i = 0; i < str_vec.size() - 1; ++i){
                res_str += str_vec[i] + pt;
        }
        res_str += *(str_vec.end() - 1);
}

std::string Utils::remove_book_mark(const std::string& str){
	const std::string& left = GlobalDict::_s_global_dict.kv_dict["LEFT_BOOK_MARK"];
        const std::string& right = GlobalDict::_s_global_dict.kv_dict["RIGHT_BOOK_MARK"];
	if (str.substr(0, left.size()) == left && 
			str.substr(str.size() - right.size(), right.size()) == right){
		return str.substr(left.size(), str.size() - left.size() - right.size());
	}
	return "";
}

void Utils::join(std::vector<std::string>::const_iterator beg,
		std::vector<std::string>::const_iterator end, std::string& res){
	res.clear();
	for (std::vector<std::string>::const_iterator iter = beg; iter != end; ++iter){
		res += *iter;
	}
}
