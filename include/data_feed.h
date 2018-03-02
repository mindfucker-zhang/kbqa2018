#ifndef _DATA_FEED_H_
#define _DATA_FEED_H_
#include <string>
#include <vector>
#include <map>
#include "tensor.h"

class DataFeed{
	
public:
	virtual int thread_init(const std::map<std::string, std::string>& conf);
	virtual void thread_destroy();
	virtual std::map<std::string, Tensor>& gen_input(void* org_data, 
			void* dict, int index = 0) = 0;
	std::vector<std::map<std::string, Tensor> >& gen_batch_input(
			std::vector<void*>& batch_org_data, void* dict);
	
protected:
	std::vector<DataType*> _memory_pool;
	std::vector<std::map<std::string, Tensor> > _batch_input;
	int _batch_size;
};

#endif
