#include <iostream>
#include <sstream>
#include <math.h>
#include "tensor.h"
#include "logger.h"
#include "sse_op.h"

Tensor::Tensor(){
	 _data = NULL;
	_memory_flag = false;
	_size = 0;
	_trans_flag = false;
}

Tensor::Tensor(const std::vector<int>& shape, bool memory_flag, DataType* data, bool trans_flag){
	_shape = shape;
	_memory_flag = memory_flag;
	_size = 1;
	_trans_flag = trans_flag;
	for (int i = 0; i < _shape.size(); ++i){
		_size *= shape[i];
	}
	if (data){
		_data = data;
	}else if (memory_flag){
		if (_trans_flag){
			_data = new DataType[_size * 2];
		}else{
			_data = new DataType[_size];
		}
		
	}else{
		_data = NULL;
	}
}

Tensor::Tensor(const int* shape, int const& dim, bool memory_flag,
		DataType* data, bool trans_flag): _shape(shape, shape + dim){
	_memory_flag = memory_flag;
	_trans_flag = trans_flag;
	_size = 1;
	for (int i = 0; i < _shape.size(); ++i){
		_size *= shape[i];
	}
	if(data){
		_data = data;
	}
	else if (memory_flag){
		if (_trans_flag){
			_data = new DataType[_size * 2];
		}else{
			_data = new DataType[_size];
		}
		
	}else{
		_data = NULL;
	}
}

Tensor::~Tensor(){
	if (_memory_flag){
		delete[] _data;
		_data = NULL;
	}
}

Tensor::Tensor(const Tensor& tensor){
	_data = tensor._data;
	_shape = tensor._shape;
	_size = tensor._size;
	_memory_flag = false;
	_trans_flag = tensor._trans_flag;
}

Tensor::Tensor(Tensor&& tensor){
	_data = tensor._data;
	_shape = tensor._shape;
	_memory_flag = tensor._memory_flag;
	_size = tensor._size;
	_trans_flag = tensor._trans_flag;
	tensor._memory_flag = false;
}

Tensor Tensor::indexing(const std::vector<int>& index){
	int pos = 0;
	if (index.size() > _shape.size()){
		Logger::logging("indexing error! (dim)", "ERROR");
		return Tensor();
	}
	
	int _cur_size = _size;
	for (int i = 0; i < index.size(); ++i){
		if (index[i] < 0 || index[i] >= _shape[i]){
			Logger::logging("indexing error!(size)", "ERROR");
			return Tensor();
		}
		_cur_size /= _shape[i];
		pos += index[i] * _cur_size;
	}

	std::vector<int> sub_shape;
	for (int i = index.size(); i < _shape.size(); ++i){
		sub_shape.push_back(_shape[i]);
	}
	Tensor ret(sub_shape, false);
	ret._data = _data + pos;
	return ret;
}

Tensor& Tensor::operator =(const Tensor& tensor){
	_size = tensor._size;
	_data = tensor._data;
	_shape = tensor._shape;
	_memory_flag = false;
	_trans_flag = tensor._trans_flag;
	return *this;	
}

Tensor& Tensor::operator +=(const Tensor& tensor){
	const std::vector<int>& add_shape = tensor._shape;
	if (add_shape.size() > _shape.size()){
		Logger::logging("tensor adding shape error !", "ERROR");
		return *this;
	}

	int size = 1;
	for (int i = 1; i <= add_shape.size(); ++i){
		if (add_shape[add_shape.size() - i] != _shape[_shape.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return *this;
		}
		size *= add_shape[add_shape.size() - i]; 
	}

	int iters = _size / size;

	if (size < 16){
		int index1 = 0;
		for (int i = 0; i < iters; ++i){
			int index2 = 0;
			for (int j = 0; j < size; ++j){
				_data[index1++] += tensor._data[index2++];	
			}
		}

	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_add(_data + (i * size), tensor._data, _data + (i * size), size);
		}
	}
	
	return *this;
}

Tensor& Tensor::operator *=(const Tensor& tensor){
	const std::vector<int>& add_shape = tensor._shape;
	if (add_shape.size() > _shape.size()){
		Logger::logging("tensor adding shape error !", "ERROR");
		return *this;
	}

	int size = 1;
	for (int i = 1; i <= add_shape.size(); ++i){
		if (add_shape[add_shape.size() - i] != _shape[_shape.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return *this;
		}
		size *= add_shape[add_shape.size() - i]; 
	}

	int iters = _size / size;

	if (size < 16){
		int index1 = 0;
		for (int i = 0; i < iters; ++i){
			int index2 = 0;
			for (int j = 0; j < size; ++j){
				_data[index1++] *= tensor._data[index2++];	
			}
		}
	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_mul(_data + (i * size), tensor._data, _data + (i * size), size);
		}
	}

	return *this;
}

const std::vector<int>& Tensor::get_shape() const{
	return _shape;
}

const DataType* Tensor::get_data() const{
	return _data;
}

int Tensor::get_size() const{
	return _size;
}

DataType* Tensor::get_var_data(){
	return _data;
}

const DataType* Tensor::get_trans_data() const{
	if (_trans_flag){
		return _data + _size;
	}else{
		return NULL;
	}
}

DataType* Tensor::get_var_trans_data(){
	if (_trans_flag){
		return _data + _size;
	}else{
		return NULL;
	}
}

std::string Tensor::to_string() const{
	std::stringstream ss;
	ss << "[";
	for (int i = 0; i < _size; ++i){
		ss << _data[i] << ",";
	}
	ss << "]";
	return ss.str();
}

std::string Tensor::to_string(int per_line) const{
	std::stringstream ss;
	int size = _shape[_shape.size() - 1];
	int iters = _size / size;
	int index = 0;
	for (int i = 0; i < iters; ++i){
		ss << "[";
		for (int j = 0; j < size; ++j){
			ss << _data[index++];
			if ((j + 1) % per_line == 0 && j != size -1){
				ss << "\n";
			}else if (j !=  size - 1){
				ss << "\t";
			}
		}
		ss << "]\n";
	}
	return ss.str();	
}


void Tensor::set_data(DataType* data, bool memory_flag){
	if (_memory_flag){
		delete[] _data;
	}
	_memory_flag = memory_flag;
	_data = data;
}

int Tensor::deep_copy(const Tensor& t){
	int size= t.get_size();
	const DataType* data = t.get_data();
	if (size != _size || !data){
		return -1;
	}
	if (!_data){
		if (!_memory_flag){
			return -1;
		}else{
			_data = new DataType[_size];
		}
	}
	for (int i = 0; i < _size; ++i){
		_data[i] = data[i];
	}
	return 0;
}

void Tensor::zero(){
	if (!_data){
		return;
	}
	for (int i = 0; i < _size; ++i){
		_data[i] = 0;
	}
}

int Tensor::transpose(int axis){
	if (!_trans_flag){
		return -1;
	}
	int dim = _shape.size();
	if (axis < 0){
		axis = dim + axis;
	}
	if (axis < 0 || axis >= dim){
		return -1;
	}
	int h1 = 1;
	for (int i = 0; i < axis; ++i){
		h1 *= _shape[i];
	}
	int h2 = _size / h1;	
	DataType* tdata = _data + _size;
	for (int i = 0; i < h1; ++i){
		for(int j = 0; j < h2; ++j){
			tdata[j * h1 + i] = _data[i * h2 + j];
		}
	}
	return 0;
}

Tensor& Tensor::operator -=(const Tensor& tensor){
	const std::vector<int>& add_shape = tensor._shape;
	if (add_shape.size() > _shape.size()){
		Logger::logging("tensor adding shape error !", "ERROR");
		return *this;
	}

	int size = 1;
	for (int i = 1; i <= add_shape.size(); ++i){
		if (add_shape[add_shape.size() - i] != _shape[_shape.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return *this;
		}
		size *= add_shape[add_shape.size() - i]; 
	}

	int iters = _size / size;

	if (size < 16){
		int index1 = 0;
		for (int i = 0; i < iters; ++i){
			int index2 = 0;
			for (int j = 0; j < size; ++j){
				_data[index1++] -= tensor._data[index2++];	
			}
		}

	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_sub(_data + (i * size), tensor._data, _data + (i * size), size);
		}
	}
	
	return *this;
}

Tensor& Tensor::operator /=(const Tensor& tensor){
	const std::vector<int>& add_shape = tensor._shape;
	if (add_shape.size() > _shape.size()){
		Logger::logging("tensor adding shape error !", "ERROR");
		return *this;
	}
	
	int size = 1;
	for (int i = 1; i <= add_shape.size(); ++i){
		if (add_shape[add_shape.size() - i] != _shape[_shape.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return *this;
		}
		size *= add_shape[add_shape.size() - i]; 
	}

	int iters = _size / size;

	if (size < 16){
		int index1 = 0;
		for (int i = 0; i < iters; ++i){
			int index2 = 0;
			for (int j = 0; j < size; ++j){
				_data[index1++] /= tensor._data[index2++];	
			}
		}

	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_div(_data + (i * size), tensor._data, _data + (i * size), size);
		}
	}
	
	return *this;
}

int Tensor::reshape(const std::vector<int>& shape){
	int size = 1;
	for (int i = 0; i < shape.size(); ++i){
		size *= shape[i];
	}
	if (size != _size){
		Logger::logging("tensor reshape error!", "ERROR");
		return -1;
	}
	_shape = shape;
}


