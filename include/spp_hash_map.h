#ifndef _SPP_HASH_MAP_
#define _SPP_HASH_MAP_
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "spp/spp.h"
#include "utils.h"
#include "logger.h"

typedef spp::sparse_hash_map<std::string, int> HashMap;

struct StringToIntSerializer{
	bool operator()(std::ofstream* stream, const std::pair<const std::string, int>& value) const
	{
		size_t sizeSecond = sizeof(value.second);
		size_t sizeFirst = value.first.size();
		stream->write((char*)&sizeFirst, sizeof(sizeFirst));
		stream->write(value.first.c_str(), sizeFirst);
		stream->write((char*)&value.second, sizeSecond);
		return true;
	}

	bool operator()(std::ifstream* istream, std::pair<const std::string, int>* value) const
	{
		size_t size = 0;
		istream->read((char*)&size, sizeof(size));
		char * first = new char[size];
		istream->read(first, size);
		new (const_cast<std::string *>(&value->first)) std::string(first, size);
		istream->read((char *)&value->second, sizeof(value->second));
		return true;
	}
};

class HashMapFileStream{

public:

	static int save_bin(const std::string& path, HashMap& hm){
		std::ofstream fo(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
		if (!fo.is_open()){
			Logger::logging("HashMap save path error: " + path, "ERROR");
			return -1;
		}
		bool tag = hm.serialize(StringToIntSerializer(), &fo);
		fo.close();
		if (!tag){
			return -1;
		}
		return 0;
	}
		
	static int load_bin(const std::string& path, HashMap& hm){
		std::ifstream fin(path.c_str());
		if (!fin.is_open()){
			Logger::logging("HashMap bin load path error: " + path, "ERROR");
			return -1;
		}
		bool tag = hm.unserialize(StringToIntSerializer(), &fin);
		fin.close();
		if (!tag){
			return -1;
		}
		return 0;
	}

	static int save_to_stream(std::ofstream& fo, HashMap& hm){
		if (!fo.is_open()){
			Logger::logging("HashMap save bin stream error!", "ERROR");
			return -1;
		}
		bool tag = hm.serialize(StringToIntSerializer(), &fo);
		if (!tag){
			return -1;
		}
		return 0;
	}
		
	static int load_from_stream(std::ifstream& fin, HashMap& hm){
		if (!fin.is_open()){
			Logger::logging("HashMap load bin stream error!", "ERROR");
			return -1;
		}
		bool tag = hm.unserialize(StringToIntSerializer(), &fin);
		if (!tag){
			return -1;
		}
		return 0;
	}

	static int load(const std::string& path, HashMap& hm, bool pair_flag = true, int rnum = 1024){
		std::ifstream fin(path.c_str());
		if (!fin.is_open()){
			Logger::logging("HashMap load path error: " + path, "ERROR");
			return -1;
		}
		hm.reserve(rnum);
		std::vector<std::string> vec;
		std::string line;
		std::stringstream ss;
		int id = 0;
		while(!fin.eof() && std::getline(fin, line)){
			if (line.size()){
				Utils::split(line, "\t", vec);
			}else{
				vec.resize(2);
				vec[0] = "";
				vec[1] = "-1";
			}
			int v = id;
			if (pair_flag){
				ss.clear();
				ss.str("");
				ss << vec[1];
				ss >> v;
			}
			hm.emplace(vec[0], v);
			++id;
		}
		return 0;
	}	

};

#endif
