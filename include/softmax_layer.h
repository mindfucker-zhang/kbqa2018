#ifndef _SOFTMAX_LAYER_
#define _SOFTMAX_LAYER_

#include "layer.h"

class SoftmaxLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);
private:
	int _axis;
	int _iters;
	int _ave_num;

};

#endif
