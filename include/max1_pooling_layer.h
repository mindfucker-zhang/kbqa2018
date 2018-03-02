#ifndef _MAX1_POOLING_LAYER_H_
#define _MAX1_POOLING_LAYER_H_

#include "layer.h"

class Max1PoolingLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs);

private:
	int _axis;

};


#endif
