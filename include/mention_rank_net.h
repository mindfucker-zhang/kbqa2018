#ifndef _MENTION_RANK_H_
#define _MENTION_RANK_H_
#include "net.h"

class MentionRankNet: public Net{

public:
	virtual const std::map<std::string, Tensor>& run(std::map<std::string,Tensor>& input, 
			std::map<std::string, std::map<std::string, Tensor> >& param_map, int index = 0);
};

#endif
