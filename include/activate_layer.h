#ifndef _ACTIVATE_LAYER_H_
#define _ACTIVATE_LAYER_H_

#include "layer.h"

class ActivateLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs);
	void set_func_key(const std::string& func_key);
private:
	std::string _func_key;
	bool _in_situ_tag;
};

#endif
