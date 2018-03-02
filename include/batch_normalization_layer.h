#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "layer.h"

class BatchNormalizationLayer: public Layer{

public:	
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
		std::map<std::string, Tensor>& params);

};

#endif 
