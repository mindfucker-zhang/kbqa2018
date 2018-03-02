#ifndef _SEMANTICS_EMBEDDING_NET_H_
#define _SEMANTICS_EMBEDDING_NET_H_

#include "net.h"

class SemanticsEmbeddingNet: public Net{

public:
	virtual const std::map<std::string, Tensor>& run(std::map<std::string,Tensor>& input, 
			std::map<std::string, std::map<std::string, Tensor> >& param_map, int index = 0);
	int get_feat_size();
};

#endif
