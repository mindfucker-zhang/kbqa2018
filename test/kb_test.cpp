#include <time.h>
#include <iostream>
#include <fstream>
#include "knowledge_base.h"

int main(int argc, char* argv[]){
	time_t t1, t2;
	time(&t1);
	int ret = KnowledgeBase::global_init(argv[1]);
	time(&t2);
	std::cout << difftime(t2, t1) << std::endl;
	//ret = KnowledgeBase::save_kb_bin_dict(argv[2]);

	/*knowledge_base_dict& dict =   KnowledgeBase::_s_kb_dict;	
	std::ofstream fo("m2e_out");
	for (auto iter = dict.mention2entity.begin(); iter !=  dict.mention2entity.end(); ++iter){
		int key = iter->first;
		std::vector<int>& vec = iter->second;
		if (vec.size() == 0){
			std::cerr << key << std::endl;
		}
		if (key != vec[0]){
			fo << key << "\t";
		}
		for (int i = 0; i < vec.size(); ++i){
			fo << vec[i];
			if (i != vec.size() - 1){
				fo << "\t";
			}else{
				fo << "\n";
			}			
		}
	}
	fo.close();

	fo.open("spo");
	for (int i = 0; i < dict.entities.size(); ++i){
		fo << "<entity_tag>\t" << dict.entities[i].entity_name << "\t" << i << "\n";
		for (int j = 0; j < dict.entities[i].po_pairs.size(); ++j){
			fo << dict.entities[i].po_pairs[j].first << "\t"
				<< dict.entities[i].po_pairs[j].second << "\n";
		}
	}

	std::string line;
	while(!std::cin.eof() && std::getline(std::cin, line)){
		std::cout << line << "\t" << dict.mention2id[line] << "\n";
	}*/

	KnowledgeBase::global_destroy();
}
