#ifndef _UTILS_H_
#define _UTILS_H_
#include <string>
#include <vector>
#include <set>
#include <map>
#include <string.h>

class Utils{
	
public:
	static void split(const std::string& str, const std::string& tag,
			std::vector<std::string>& res);
	static std::vector<std::string> split(const std::string& str, const std::string& tag);
	static float str2float(const std::string& str);
	static int str2int(const std::string& str);
	static bool is_empty(const std::string& str);
	static int read_conf_map(const std::string& conf_file,
        		std::map<std::string, std::string>& conf_map);
	static int load_dict(const std::string& path, std::set<std::string>& res);
	static int load_dict(const std::string& path, std::map<std::string, int>& res, 
			const std::string& pt = "\t");
	static int load_dict(const std::string& path, std::map<std::string, float>& res,
			const std::string& pt = "\t");
	static int word2char_gbk(const std::string& word, std::vector<std::string>& char_vec);
	static void merge_vector(const std::vector<std::vector<std::string> >& vec,
			std::vector<std::string>& res);
	static void join(const std::vector<std::string> &str_vec, const std::string &pt,
			std::string &res_str);
	static std::string remove_book_mark(const std::string& str);
	static void join(std::vector<std::string>::const_iterator beg,
			std::vector<std::string>::const_iterator end, std::string& res);

};

#endif
