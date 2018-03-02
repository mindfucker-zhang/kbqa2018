#ifndef _LSTM_LAYER_H_
#define _LSTM_LAYER_H_
#include "layer.h"

class LstmLayer: public Layer{

public:
	virtual int thread_init();
	virtual Tensor process(std::map<std::string, Tensor>& inputs,
			std::map<std::string, Tensor>& params);
	void set_reversed(bool flag);

private:
	bool _reversed_flag;
	int _seq_len;
};

#endif
