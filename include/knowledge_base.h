#ifndef _KNOWLEDGE_BASE_H_
#define _KNOWLEDGE_BASE_H_
#include <map>
#include <vector>
#include <string>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include "spp_hash_map.h"

struct entity_kb_node{
	std::string entity_name;
	std::vector<std::pair<std::string, std::string> > po_pairs;

        template<typename Archive>
        void serialize(Archive& ar, const unsigned int version){
		ar & entity_name;
		ar & po_pairs; 
	}
};

struct knowledge_base_dict{
	HashMap mention2id;	
	std::map<int, std::vector<int> > mention2entity;
	std::vector<entity_kb_node> entities;
	int entity_num;

	void clear(){
		mention2id.clear();
		mention2entity.clear();
		entities.clear();
		entity_num = 0;
	}
};


class KnowledgeBase{

public:
	int thread_init();
	void thread_destroy();
	int get_mention_id(const std::string& mention);
	std::vector<std::pair<int, std::string> > get_entities_by_mention(
			const std::string& mention);
	const entity_kb_node* get_entity_info_by_id(int eid);

private:
	static knowledge_base_dict  _s_kb_dict;

public:
	static int  global_init(const std::string& conf_path); 
	static void global_destroy();
	static int load_kb_dict(const std::map<std::string, std::string>& conf);
	static int load_kb_bin_dict(const std::string& bin_path);
	static int load_mention2id(const std::string& dict_path, int num);
	static int load_mention2entity(const std::string& dict_path);
	static int load_entities(const std::string& dict_path);
	static int save_kb_bin_dict(const std::string& bin_path);
	
};


#endif
