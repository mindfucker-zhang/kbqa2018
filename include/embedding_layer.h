#ifndef _EMBEDDING_LAYER_
#define _EMBEDDING_LAYER_

#include "layer.h"

class EmbeddingLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);
};

#endif
