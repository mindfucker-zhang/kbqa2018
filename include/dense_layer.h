#ifndef _DENSE_LAYER_H_
#define _DENSE_LAYER_H_

#include "layer.h"

class DenseLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);

	void set_bias_flag(bool bias_flag);

private:
	bool _bias_flag;	

};


#endif
