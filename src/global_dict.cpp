#include "utils.h"
#include "logger.h"
#include "global_dict.h"

global_dict GlobalDict::_s_global_dict;

int GlobalDict::global_init(const std::string& conf_path){
	_s_global_dict.stops.clear();
	_s_global_dict.kv_dict.clear();

	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
		Logger::logging("global dict conf path error: " + conf_path, "ERROR");
		return -1;
	}

	if (conf_map.find("stopwords") == conf_map.end()
	    ||  conf_map.find("kv_dict") == conf_map.end()){
		Logger::logging("cannot find stopwords or kv_dict!", "ERROR");
		return -1;
	}

	if (Utils::load_dict(conf_map["stopwords"], _s_global_dict.stops) != 0){
		Logger::logging("cannot load stopwords!", "ERROR");
		return -1;
	}

	if (Utils::read_conf_map(conf_map["kv_dict"], _s_global_dict.kv_dict) != 0){
		Logger::logging("cannot load kv_dict!", "ERROR");
		return -1;
	}
	return 0;	
}

void GlobalDict::global_destroy(){
	_s_global_dict.stops.clear();
	_s_global_dict.kv_dict.clear();
}
