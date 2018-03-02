#include <iostream>
#include <fstream>
#include <string.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include "net.h"
#include "dense_layer.h"
#include "embedding_layer.h"
#include "lstm_layer.h"
#include "softmax_layer.h"
#include "concat_layer.h"
#include "activate_layer.h"
#include "batch_normalization_layer.h"
#include "conv1d_layer.h"
#include "max1_pooling_layer.h"
#include "utils.h"
#include "logger.h"

Net::~Net(){
	thread_destroy();
}

Layer* Net::layer_malloc_by_key(const std::string& key){
	std::string layer_type;
	for (int i = key.size() - 1; i != -1; --i){
		if (key[i] == '_'){
			layer_type = key.substr(i + 1, key.size() - i - 1);
			break;
		}
	}
	if (layer_type == ""){
		layer_type = key;
	}

	if (layer_type == "dense"){
		return new DenseLayer();	
	}
	if (layer_type == "lstm"){
		return new LstmLayer();
	}
	if (layer_type == "embedding"){
		return new EmbeddingLayer();
	}
	if (layer_type == "softmax"){
		return new SoftmaxLayer();
	}
	if (layer_type == "concat"){
		return new ConcatLayer();
	}
	if (layer_type == "activate"){
		return new ActivateLayer();
	}
	if (layer_type == "bn"){
		return new BatchNormalizationLayer();
	}
	if (layer_type == "conv1d"){
		return new Conv1dLayer();
	}
	if (layer_type == "max1pool"){
		return new Max1PoolingLayer();
	}
	return NULL;
}

int Net::thread_init(const std::map<std::string, std::map<std::string,
		std::vector<int> > >& conf_map, int batch_size){
	_batch_size = batch_size;
	_layer_maps.resize(batch_size);
	for (int i = 0; i < _batch_size; ++i){
		std::map<std::string, std::map<std::string, 
			std::vector<int> > >::const_iterator iter = conf_map.begin();
		for (; iter != conf_map.end(); ++iter){
			const std::string& layer_key = iter->first; 
			const std::map<std::string, std::vector<int> >& param_conf = iter->second;
			Layer* layer = layer_malloc_by_key(layer_key);		
			layer->set_param_conf(param_conf);
			if (layer->thread_init() != 0){
				Logger::logging(layer_key + " thread_init error!", "ERROR");
				delete layer;
				return -1;
			}
			_layer_maps[i][layer_key] = layer;
		}
	}
	_batch_results.resize(_batch_size);
	_pthreads = new pthread_t[_batch_size];
	_thread_inputs = new net_thread_data[_batch_size];
	return 0;	
}

void Net::thread_destroy(){
	for (int i = 0; i < _layer_maps.size(); ++i){
		std::map<std::string, Layer*>::iterator iter = 
				_layer_maps[i].begin();
		for (; iter != _layer_maps[i].end(); ++iter){
			iter->second->thread_destroy();
			delete iter->second;
		}
	}
	_layer_maps.clear();
	_batch_results.clear();
	delete[] _pthreads;
	_pthreads = NULL;
	delete[] _thread_inputs;
	_thread_inputs = NULL;
}

const std::vector<std::map<std::string, Tensor> >& Net::run_batch(std::vector<std::map<
		std::string, Tensor> >& batch_inputs, std::map<std::string, std::map<
		std::string, Tensor> >& param_map){
	int batch_size = _batch_size;
	if (batch_inputs.size() < batch_size){
		batch_size = batch_inputs.size();
	}
	for (int i = 0; i < batch_size; ++i){
		_thread_inputs[i].input = &batch_inputs[i];
		_thread_inputs[i].param_map = &param_map;
		_thread_inputs[i].index = i;
		_thread_inputs[i].net = this;	
		if (pthread_create(&_pthreads[i], 0, run_net_thread, 
				(void*) &_thread_inputs[i]) != 0){
			Logger::logging("thread create error !", "ERROR");
			return _batch_results;
		}
	}
	for (int i = 0; i < batch_size; ++i){
		pthread_join(_pthreads[i], NULL);
	}
	return _batch_results;
}

void* run_net_thread(void* thread_input){
	net_thread_data* input_ptr = (net_thread_data*)thread_input;
	Net* net = input_ptr->net;
	net->run(*(input_ptr->input), *(input_ptr->param_map), input_ptr->index);
}

int Net::load_para_conf(const std::string& conf_path, std::map<std::string,
	std::map<std::string, std::vector<int> > >& para_conf){
	std::ifstream fin(conf_path.c_str());
	if (!fin.is_open()){
		Logger::logging("para conf file error: " + conf_path, "ERROR");
		return -1;
	}
	std::string line;
	while(!fin.eof() && std::getline(fin, line)){
		if (!line.size() || line[0] == '#'){
			continue;
		}
		std::vector<std::string> kv = Utils::split(line, ": ");
		if (kv.size() != 2){
			continue;
		}
		std::vector<std::string> key = Utils::split(kv[0], "/");
		if (key.size() != 2){
			continue;
		}
		std::vector<int> value;
		if (kv[1].size() < 2){
			continue;
		}
		std::vector<std::string> str_value = Utils::split(
				kv[1].substr(1, kv[1].size() -2), ", ");
		for (int i = 0; i < str_value.size(); ++i){
			value.push_back(Utils::str2int(str_value[i]));
		}
		if (para_conf.find(key[0]) == para_conf.end()){
			para_conf[key[0]] = std::map<std::string, std::vector<int> >();
		}
		para_conf[key[0]][key[1]] = value;
	}
	return 0;
}

int Net::load_bin_model(const std::string& model_path, net_model& model){
	std::ifstream fin(model_path.c_str());
	if (!fin.is_open()){
		Logger::logging("model bin file error: " + model_path, "ERROR");
		return -1;
	} 
	boost::archive::binary_iarchive bin_fin(fin);
	bin_fin >> model;
	fin.close();
	return 0;
}

int Net::save_bin_model(const std::string& model_path, net_model& model){
	std::ofstream fo(model_path.c_str());
	if (!fo.is_open()){
		Logger::logging("model path file error: " + model_path, "ERROR");
		return -1;
	}
	boost::archive::binary_oarchive bin_fo(fo);
	bin_fo << model;
	fo.close();
	return 0;
}

void Net::destroy_model(net_model& model){
	for (int i = 0; i < model.memory_pool.size(); ++i){
		if (model.memory_pool[i]){
			delete[] model.memory_pool[i];
		}
	}
	model.memory_pool.clear();
	model.params.clear();
}

int Net::load_model(const std::string& model_path, net_model& model){
	destroy_model(model);
	std::ifstream fin(model_path.c_str());
	if (!fin.is_open()){
			Logger::logging("model file error: " + model_path, "ERROR");
			return -1;
	}
	std::vector<DataType*>& memory_pool = model.memory_pool;
	std::map<std::string, std::map<std::string, Tensor> >& params = model.params;
	std::string line;
	while(!fin.eof() && std::getline(fin, line)){
		if (!line.size() || line[0] == '#' || Utils::is_empty(line)){
			continue;
		}
		std::vector<std::string> kv = Utils::split(line, ": ");
		if (kv.size() != 2){
			continue;
		}
		std::vector<std::string> key = Utils::split(kv[0], "/");
		if (key.size() != 2){
			continue;
		}
		if (kv[1].size() < 2){
			continue;
		}
		std::vector<int> shape;
		int size = 1;
		std::vector<std::string> str_value = Utils::split(
				kv[1].substr(1, kv[1].size() -2), ", ");
		for (int i = 0; i < str_value.size(); ++i){
			int value = Utils::str2int(str_value[i]);
			shape.push_back(value);
			size *= value;
		}
		if (params.find(key[0]) == params.end()){
			params[key[0]] = std::map<std::string, Tensor>();
		}
		DataType* data = new DataType[size * 2];
		for (int i = 0; i < size; ++i){
			fin >> data[i];
		}
		Tensor tensor(shape, false, data, true);
		tensor.transpose();
		params[key[0]][key[1]] = tensor;
		memory_pool.push_back(data);
	}
	return 0;
}

net_dict* Net::global_init(const std::string& conf_path){
	std::map<std::string, std::string> conf_map;
	if (Utils::read_conf_map(conf_path, conf_map) != 0){
			Logger::logging("net dict read conf map error!", "ERROR");
			return NULL;
	}
	net_dict* dict = new net_dict();
	char* keys[] = {"model_type", "net_model_path", "net_para_conf", "batch_size"};
	int key_num = 4;
	int ret = 0;
	std::string model_type;
	for (int i = 0; i < key_num; ++i){
		auto iter = conf_map.find(keys[i]);
		if (iter == conf_map.end()){
			Logger::logging(std::string(keys[i]) + "not in conf map!", "ERROR");
			ret = -1;
			break;
		}
		if (strcmp(keys[i], "net_model_path") == 0){
			if (conf_map["model_type"] == "bin"){
				ret = load_bin_model(iter->second, dict->model);
			}else{
				ret = load_model(iter->second, dict->model);
			}
		}else if (strcmp(keys[i], "net_para_conf") == 0){
			ret = load_para_conf(iter->second, dict->para_conf);
		}else if (strcmp(keys[i], "batch_size") == 0){
			dict->batch_size = Utils::str2int(iter->second);
		}
		if (ret != 0){
			Logger::logging(std::string(keys[i]) + " load error!", "ERROR");
			break;
		}
	}
	
	if (ret != 0){
		destroy_model(dict->model);
		delete dict;
		return NULL;
	}
	return dict;
}

void Net::global_destroy(net_dict*& dict){
	if (dict){
		destroy_model(dict->model);
		delete dict;
		dict = NULL;
	}
}

int Net::get_output_shape(const std::string& key, std::vector<int>& res_shape){
	if (!_batch_size){
		Logger::logging("net batch is empty!", "ERROR");
		return -1;
	}
	auto iter = _batch_results[0].find(key);
	if (iter == _batch_results[0].end()){
		Logger::logging("Net::get_output_shape: key error!", "ERROR");
		return -1;
	}
	res_shape = iter->second.get_shape();
	return 0;
}
