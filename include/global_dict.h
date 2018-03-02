#ifndef _QASYS_GLOBAL_DICT_H_
#define _QASYS_GLOBAL_DICT_H_
#include <map>
#include <string>
#include <set>

struct global_dict{
	std::set<std::string> stops;
        std::map<std::string, std::string> kv_dict;
};

class GlobalDict{
public:
	static int global_init(const std::string& conf_path);
	static void global_destroy();
public:	
	static global_dict _s_global_dict;	
};

#endif
