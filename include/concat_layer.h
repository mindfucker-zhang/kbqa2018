#ifndef _CONCAT_LAYER_
#define _CONCAT_LAYER_
#include "layer.h"

class ConcatLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs);

private:
	int _axis;	
	
};

#endif
