#include <fstream>
#include <utility>
#include <time.h>
#include "knowledge_base.h"
#include "utils.h"
#include "logger.h"

knowledge_base_dict KnowledgeBase::_s_kb_dict;

int KnowledgeBase::load_mention2id(const std::string& dict_path, int num){
	int ret = HashMapFileStream::load(dict_path, _s_kb_dict.mention2id, 
			true, num);
	if (ret != 0){
		Logger::logging("knowledge base mention2id load error: " + dict_path, "ERROR");
	}
	return ret;
};

int KnowledgeBase::load_mention2entity(const std::string& dict_path){
	std::ifstream fin(dict_path.c_str());
	if (!fin.is_open()){
		Logger::logging("knowledge base mentiont2entity path error: "
				+ dict_path,"ERROR");
		return -1;
	}

	std::string line;
	std::vector<std::string> str_ids;
	while(!fin.eof() && std::getline(fin, line)){
		Utils::split(line, "\t", str_ids);
		if (str_ids.size() < 1){
			continue;
		}
		int key = Utils::str2int(str_ids[0]);
		auto iter = _s_kb_dict.mention2entity.find(key);
		if (iter == _s_kb_dict.mention2entity.end()){
			iter = _s_kb_dict.mention2entity.insert(
					std::make_pair(key, std::vector<int>())).first;
		}
		std::vector<int>& ids = iter->second;
		int size = ids.size() + str_ids.size();
		int st = 0;
		if (key >= _s_kb_dict.entity_num){
			--size;
			st = 1;
		}
		ids.reserve(size);
		for (int i = st; i < str_ids.size(); ++i){
			ids.push_back(Utils::str2int(str_ids[i]));
		}
			
	}
	fin.close();
	return 0;
}

int KnowledgeBase::load_entities(const std::string& dict_path){
	std::ifstream fin(dict_path.c_str());
        if (!fin.is_open()){
                Logger::logging("knowledge entities  path error: "
                                + dict_path, "ERROR");
                return -1;
        }

	_s_kb_dict.entities.resize(_s_kb_dict.entity_num);
	std::string line;
	std::vector<std::string> vec;
	int cur_id = -1;
	while(!fin.eof() && std::getline(fin, line)){
		Utils::split(line, "\t", vec);
		if (vec[0] == "<entity_tag>"){
			if (vec.size() < 3){
				Logger::logging("entitiy format error: " + line, "ERROR");
				continue;
			}
			int id = Utils::str2int(vec[2]);
			if (id >= _s_kb_dict.entity_num || id < 0){
				Logger::logging("entitiy id error: " + line, "ERROR");
				continue;
			}
			cur_id = id;
			_s_kb_dict.entities[id].entity_name = vec[1];
			continue;
		}
	
		if (cur_id == -1){
			continue;
		}
		_s_kb_dict.entities[cur_id].po_pairs.push_back(std::make_pair(vec[0], vec[1]));
	}
	fin.close();
	return 0;
}

int KnowledgeBase::load_kb_dict(const std::map<std::string, std::string>& conf){
	char* keys[] = {"entity_num", "mention_num", "mention2id", "mention2entity", "entities"};
	int key_num = 5;
	int mention_num = 0;
	int ret = 0;
	for (int i = 0; i < key_num; ++i){
		auto iter = conf.find(keys[i]);
		if (iter == conf.end()){
			Logger::logging("cannot find " + std::string(keys[i]) + 
					" in kb_conf", "ERROR");
			ret = -1;
			break;
		}
		if (strcmp(keys[i], "entity_num") == 0){
			_s_kb_dict.entity_num = Utils::str2int(iter->second);
		}else if (strcmp(keys[i], "mention_num") == 0){
			mention_num = Utils::str2int(iter->second);
		}else if (strcmp(keys[i], "mention2id") == 0){
			ret = load_mention2id(iter->second, mention_num);
		}else if (strcmp(keys[i], "mention2entity") == 0){
			ret = load_mention2entity(iter->second);
		}else if (strcmp(keys[i], "entities") == 0){
			ret = load_entities(iter->second);
		}
		
		if (ret != 0){
			Logger::logging(std::string(keys[i]) + " load error!", "ERROR");
			_s_kb_dict.clear();
			break;
		}
	}
	return ret;
}

int KnowledgeBase::load_kb_bin_dict(const std::string& bin_path){
	std::ifstream fin(bin_path.c_str());
	if (!fin.is_open()){
		Logger::logging("cannot load kb bin dict: " + bin_path, "ERROR");
		return -1;
	}
	
	int ret = HashMapFileStream::load_from_stream(fin, _s_kb_dict.mention2id);
	if (ret != 0){
		Logger::logging("load mention2id bin error!", "ERROR");
		return -1;
	}

	boost::archive::binary_iarchive bin_fin(fin);
	bin_fin >> _s_kb_dict.entity_num;
	bin_fin >> _s_kb_dict.mention2entity;
	bin_fin >> _s_kb_dict.entities;
	fin.close();
	return 0;
}

int KnowledgeBase::save_kb_bin_dict(const std::string& bin_path){
	std::ofstream fo(bin_path.c_str());
	if (!fo.is_open()){
		Logger::logging("cannot save kb bin dict: " + bin_path, "ERROR");
		return -1;
	}

	int ret = HashMapFileStream::save_to_stream(fo, _s_kb_dict.mention2id);
	if (ret != 0){
		Logger::logging("save mention2id bin error!", "ERROR");
		return -1;
	}
	boost::archive::binary_oarchive bin_fo(fo);
	bin_fo << _s_kb_dict.entity_num;
	bin_fo << _s_kb_dict.mention2entity;
	//_s_kb_dict.entities.reserve(_s_kb_dict.entity_num);
	bin_fo << _s_kb_dict.entities;
	fo.close();
	return 0;
}

int KnowledgeBase::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	int ret = Utils::read_conf_map(conf_path, conf_map);
	if (ret != 0){
		Logger::logging("knowledge base conf load error: " + conf_path, "ERROR");
		return -1;	
	}
	if (conf_map.find("type") == conf_map.end()){
		Logger::logging("cannot find type in  conf!", "ERROR");
		return -1;
	}
	if (conf_map["type"] == "bin"){
		if (conf_map.find("bin_path") == conf_map.end()){
			Logger::logging("cannot find bin path in conf!", "ERROR");
			return -1;
		}
		ret = load_kb_bin_dict(conf_map["bin_path"]);
		if (ret != 0){
			Logger::logging("load knowledge base bin dict error!", "ERROR");
			return -1;
		}
		return 0;
	}

	ret = load_kb_dict(conf_map);
	if (ret != 0){
		Logger::logging("load knowledge base dict error!", "ERROR");
		return -1;
	}
	return 0;
}

void KnowledgeBase::global_destroy(){
	_s_kb_dict.clear();
}

int KnowledgeBase::thread_init(){
	return 0;
}

void KnowledgeBase::thread_destroy(){

}

int KnowledgeBase::get_mention_id(const std::string& mention){
	auto iter = _s_kb_dict.mention2id.find(mention);
	if (iter == _s_kb_dict.mention2id.end()){
		return -1;
	}
	return iter->second ;
}

std::vector<std::pair<int, std::string> > KnowledgeBase::get_entities_by_mention(
		const std::string& mention){
	std::vector<std::pair<int, std::string> > ret;
	auto iter = _s_kb_dict.mention2id.find(mention);
	if (iter == _s_kb_dict.mention2id.end()){
		return ret;
	}
	int id = iter->second;
	auto eiter = _s_kb_dict.mention2entity.find(id);
	if (eiter == _s_kb_dict.mention2entity.end()){
		if (id < _s_kb_dict.entity_num){
			ret.push_back(std::make_pair(id, _s_kb_dict.entities[id].entity_name));
		}
		return ret;
	}

	std::vector<int>& eids = eiter->second;
	for (int i = 0; i < eids.size(); ++ i){
		ret.push_back(std::make_pair(eids[i], _s_kb_dict.entities[eids[i]].entity_name));
	}
	return ret;
}

const entity_kb_node* KnowledgeBase::get_entity_info_by_id(int eid){
	if (eid >= _s_kb_dict.entity_num){
		return NULL;
	}
	return &_s_kb_dict.entities[eid];
}
