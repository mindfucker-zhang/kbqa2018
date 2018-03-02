#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include "data_type.h"

class Tensor{

public:
	Tensor();
	Tensor(const std::vector<int>& shape, bool memory_flag = false, 
			DataType* data = NULL, bool trans_flag = false);
	Tensor(const Tensor& tensor);
	Tensor(Tensor&& tensor);
	Tensor(const int* shape, int const& dim, bool memory_flag = false,
			DataType* data = NULL, bool trans_flag = false);
	~Tensor();
	Tensor indexing(const std::vector<int>& index);
	int deep_copy(const Tensor& t);

	const std::vector<int>& get_shape() const;
	const DataType* get_data() const;
	int get_size() const;
	DataType* get_var_data();

	const DataType* get_trans_data() const;
	DataType* get_var_trans_data();

	int reshape(const std::vector<int>& shape);
	void set_data(DataType* data, bool memory_flag = false);	
	void zero();
	int transpose(int aixs = -1);
	std::string to_string() const;
	std::string to_string(int per_line) const;

	Tensor& operator +=(const Tensor& tensor);
	Tensor& operator *=(const Tensor& tensor);
	Tensor& operator -=(const Tensor& tensor);
	Tensor& operator /=(const Tensor& tensor);
	Tensor& operator =(const Tensor& tensor);

private:
	friend class boost::serialization::access;
	/*template<typename Archive>
	void serialize(Archive& ar, const unsigned int version){
		ar & _size;
		ar &  _memory_flag;
		ar & _shape;
		for (int i = 0; i < _size; ++i){
			ar & _data[i];
		}
	}*/

        void serialize(boost::archive::binary_oarchive& ar, const unsigned int version){
                ar & _size;
                ar &  _memory_flag; 
                ar & _shape;
		ar & _trans_flag;
		int size = _size;
		if (_trans_flag){
			size *= 2;
		}
                for (int i = 0; i < size; ++i){
                        ar & _data[i];
                } 
        }	

	void serialize(boost::archive::binary_iarchive& ar, const unsigned int version){
                ar & _size;
                ar &  _memory_flag;
                ar & _shape;
		ar & _trans_flag;
		int size = _size;
		if (_trans_flag){
			size *= 2;
		}
		_data = new DataType[size];
                for (int i = 0; i < size; ++i){
                        ar & _data[i];
                }
        }

private:
	DataType* _data;
	std::vector<int> _shape;
	int _size;
	bool _memory_flag;
	bool _trans_flag;
};



#endif 

