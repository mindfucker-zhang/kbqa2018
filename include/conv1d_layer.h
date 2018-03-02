#ifndef _CONV1D_LAYER_H_
#define _CONV1D_LAYER_H_
#include "layer.h"

class Conv1dLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);
	void set_bias_flag(bool bias_flag);
private:
	int _iters;
	int _win_size;
	int _in_seq_len;
	int _in_feat_num;
	int _out_feat_num;
	bool _bias_flag;
};

#endif
