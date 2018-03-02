#ifndef _NET_H_
#define _NET_H_
#include <map>
#include <vector>
#include <string>
#include <pthread.h>
#include "layer.h"

class Net;

struct net_thread_data{
	Net* net;
	std::map<std::string, Tensor>* input;
	std::map<std::string, std::map<std::string, Tensor> >* param_map;
	int index;

	net_thread_data(){
		input = NULL;
		param_map = NULL;
		index = 0;
	}
};

struct net_model{
	std::map<std::string, std::map<std::string, Tensor> > params;
	std::vector<DataType*>  memory_pool;

	friend class boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version){
		ar & params;
		memory_pool.clear();
		std::map<std::string, std::map<std::string,
				Tensor> >::iterator iter1 = params.begin();
		for (; iter1 != params.end(); ++ iter1){
			std::map<std::string, Tensor>::iterator iter2 = iter1->second.begin();
			for (; iter2 != iter1->second.end(); ++iter2){
				memory_pool.push_back(iter2->second.get_var_data());
			}
		}
	}
};

void* run_net_thread(void* thread_input);

struct net_dict{
	net_model model;
	std::map<std::string, std::map<std::string, std::vector<int> > > para_conf;
	int batch_size;
};

class Net{

public:
	virtual ~Net();
	int thread_init(const std::map<std::string, std::map<std::string, 
			std::vector<int> > >& conf_map, int batch_size);
	void thread_destroy();
	const std::vector<std::map<std::string, Tensor> >& run_batch(std::vector<std::map<std::string, 
			Tensor> >& batch_inputs, std::map<std::string, std::map<std::string, Tensor> >& param_map);
	virtual const std::map<std::string, Tensor>& run(std::map<std::string, Tensor>& input,
			std::map<std::string, std::map<std::string, Tensor> >& param_map, int index = 0) = 0;
	int get_output_shape(const std::string& key, std::vector<int>& res_shape);

protected:
	Layer* layer_malloc_by_key(const std::string& key);

protected:
	std::vector<std::map<std::string, Layer*> > _layer_maps;
	std::vector<std::map<std::string, Tensor> > _batch_results;
	int _batch_size;
	pthread_t* _pthreads;
	net_thread_data* _thread_inputs;

public:
	static int load_para_conf(const std::string& conf_path, std::map<std::string,
			std::map<std::string, std::vector<int> > >& para_conf);
	static int load_bin_model(const std::string& model_path, net_model& model);
	static int load_model(const std::string& model_path, net_model& model);
	static int save_bin_model(const std::string& model_path, net_model& model);
	static void destroy_model(net_model& model);
	static net_dict* global_init(const std::string& conf_path);
	static void global_destroy(net_dict*& dict);

};


#endif
