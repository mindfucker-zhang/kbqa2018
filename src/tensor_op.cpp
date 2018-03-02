#include <iostream>
#include <math.h>
#include "tensor_op.h"
#include "logger.h"
#include "sse_op.h"

Tensor TensorOperation::add(const Tensor& t1, const Tensor& t2, DataType* data){
	const Tensor* ptr1;
	const Tensor* ptr2;
	if (t1.get_shape().size() >= t2.get_shape().size()){
		ptr1 = &t1;
		ptr2 = &t2;
	}else{
		ptr1 = &t2;
		ptr2 = &t1;
	}
	const std::vector<int>& shape1 = ptr1->get_shape();
	const std::vector<int>& shape2 = ptr2->get_shape();
	const DataType* data1 = ptr1->get_data();
	const DataType* data2 = ptr2->get_data();
	
	int size = 1;
	for (int i = 1; i <= shape2.size(); ++i){
		if (shape2[shape2.size() - i] != shape1[shape1.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return Tensor();
		}
		size *= shape2[shape2.size() - i]; 
	}

	int iters = ptr1->get_size() / size;

	Tensor ret(shape1, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}

	DataType* ret_data = ret.get_var_data();	

	if (size < 16){
		for (int i = 0; i < iters; ++i){
			for (int j = 0; j < size; ++j){
				ret_data[i * size + j] = data1[i * size + j] + data2[j];
			}
		}
	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_add(data1 + (i *size) , data2, ret_data + (i * size), size); 
		}
	}
	return ret;
}

Tensor TensorOperation::mul(const Tensor& t1, const Tensor& t2, DataType* data){
	const Tensor* ptr1;
	const Tensor* ptr2;
	if (t1.get_shape().size() >= t2.get_shape().size()){
		ptr1 = &t1;
		ptr2 = &t2;
	}else{
		ptr1 = &t2;
		ptr2 = &t1;
	}
	const std::vector<int>& shape1 = ptr1->get_shape();
	const std::vector<int>& shape2 = ptr2->get_shape();
	const DataType* data1 = ptr1->get_data();
	const DataType* data2 = ptr2->get_data();
	
	int size = 1;
	for (int i = 1; i <= shape2.size(); ++i){
		if (shape2[shape2.size() - i] != shape1[shape1.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return Tensor();
		}
		size *= shape2[shape2.size() - i]; 
	}

	int iters = ptr1->get_size() / size;

	Tensor ret(shape1, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}

	DataType* ret_data = ret.get_var_data();	

	if (size < 16){
		for (int i = 0; i < iters; ++i){
			for (int j = 0; j < size; ++j){
				ret_data[i * size + j] = data1[i * size + j] * data2[j];
			}
		}
	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_mul(data1 + (i *size) , data2, ret_data + (i * size), size);
		}
	}
		
	return ret;
}

Tensor TensorOperation::matmul(const Tensor& t1, const Tensor& t2, DataType* data){
	const std::vector<int>& shape1 = t1.get_shape();	
	const std::vector<int>& shape2 = t2.get_shape();
	int dim1 = shape1.size();
	int dim2 = shape2.size();
	if (dim1 * dim2 == 0){
		Logger::logging("matmul shape error!", "ERROR");
		return Tensor();
	}
	if (shape1[dim1 -1] != shape2[0]){
		Logger::logging("matmul shape error!", "ERROR");
		return Tensor();
	}

	int d = shape2[0]; 
	int h1 = t1.get_size() / d;
	int h2 = t2.get_size() / d;
	std::vector<int> ret_shape;
	for (int i = 0; i < dim1 - 1; ++i){
		ret_shape.push_back(shape1[i]);
	}
	for (int i = 1; i < dim2; ++i){
		ret_shape.push_back(shape2[i]);
	}

	Tensor ret(ret_shape, false);
	if (data){
		ret.set_data(data, false);	
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}
	
	const DataType* data1 = t1.get_data();
	const DataType* data2 = t2.get_data();
	DataType* ret_data = ret.get_var_data();

	if (d < 16){
		for (int i = 0; i < h1; ++i){
			for (int j = 0; j < h2; ++j){
				int index = i * h2 + j;
				ret_data[index] = 0.0;
				for (int k = 0; k < d; ++k){
					ret_data[index] += data1[i * d + k] * data2[k * h2 + j];
				}
			}
		}
	}else{
		const DataType* trans_data2 = t2.get_trans_data();
		if (!trans_data2){
			DataType* buff = new DataType[d];
     	   		DataType* ret_buff = new DataType[d];
			for (int i = 0; i < h1; ++i){
				for (int j = 0; j < h2; ++j){
					int index = i * h2 + j;
					for (int k = 0; k < d; ++k){
						buff[k] = data2[k * h2 + j];
					}
				
					sse_vector_mul(data1 + (i * d), buff, ret_buff, d);
					sse_vector_sum(ret_buff, d, ret_data[index]);
				}
			}
			delete[] buff;
			delete[] ret_buff;
		}else{
			DataType* ret_buff = new DataType[d];
			for (int i = 0; i < h1; ++i){
				for (int j = 0; j < h2; ++j){
					int index = i * h2 + j;
					sse_vector_mul(data1 + (i * d), trans_data2 + (j * d), ret_buff, d);
					sse_vector_sum(ret_buff, d, ret_data[index]);
				}
			}
			delete[] ret_buff;
		}
	}
	return ret;
}

Tensor TensorOperation::concat(const Tensor& t1, const Tensor& t2, int axis, DataType* data){
	const std::vector<int>& shape1 = t1.get_shape();
	const std::vector<int>& shape2 = t2.get_shape();
	int dim1 = shape1.size();
	int dim2 = shape2.size();
	if (dim1 * dim2 == 0 || dim1 != dim2 || axis < 0 || axis >= dim1){
		Logger::logging("concat shape error!", "ERROR");
		return Tensor();
	}

	int h1 = 1;
	int h2 = 1;
	std::vector<int> ret_shape;
	for (int i = 0; i < axis; ++i){
		if (shape1[i] != shape2[i]){
			Logger::logging("concat shape error!", "ERROR");
			return Tensor();
		}
		ret_shape.push_back(shape1[i]);
		h1 *= shape1[i];
	}
	ret_shape.push_back(shape1[axis] + shape2[axis]);
	for (int i = axis + 1; i < dim1; ++i){
		if (shape1[i] != shape2[i]){
			Logger::logging("concat shape error!", "ERROR");
			return Tensor();
		}
		ret_shape.push_back(shape1[i]);
		h2 *= shape2[i];
	}

	Tensor ret(ret_shape, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}
	
	const DataType* data1 = t1.get_data();
	const DataType* data2 = t2.get_data();
	DataType* ret_data = ret.get_var_data();
	int index = 0;
	int index1 = 0;
	int index2 = 0;
	for (int i = 0; i < h1; ++i){
		for (int j = 0; j < shape1[axis] * h2; ++j){
			ret_data[index++] = data1[index1++];
		}
		for (int j = 0; j < shape2[axis] * h2; ++j){
			ret_data[index++] = data2[index2++];
		}	
	}
	return ret;
}

Tensor TensorOperation::activation_func(const Tensor& t1, const std::string& type, DataType* data){
	const std::vector<int>& shape = t1.get_shape();
	const DataType* src_data = t1.get_data();
	int size = t1.get_size();
	Tensor ret(shape, false);
        if (data){
                ret.set_data(data, false);
        }else{
                DataType* buff = new DataType[size];
                ret.set_data(buff, true);
        }
	DataType* ret_data = ret.get_var_data();

	if (type == "sigmoid"){
		sigmoid(src_data, ret_data, size);
	}else if (type == "tanh"){
		tanh(src_data, ret_data, size);
	}else if (type == "relu"){
		relu(src_data, ret_data, size);
	}
	return ret;
}

void TensorOperation::activation_func_in_situ(Tensor& t1, const std::string& type){
	const std::vector<int>& shape = t1.get_shape();
	DataType* data = t1.get_var_data();
	int size = t1.get_size();

	if (type == "sigmoid"){
		sigmoid(data, data, size);
	}else if (type == "tanh"){
		tanh(data, data, size);
	}else if (type == "relu"){
		relu(data, data, size);
	}
}

void TensorOperation::sigmoid(const DataType* data1, DataType* data2, int size){
	for (int i = 0; i < size; ++i){
		data2[i] = 1.0 / (exp(-data1[i]) + 1.0);
	}
}

void TensorOperation::relu(const DataType* data1, DataType* data2, int size){
	for (int i = 0; i < size; ++i){
		if (data1[i] > 0){
			data2[i] = data2[i];
		}
		else{
			data2[i] = 0;
		}
	}
}

void TensorOperation::tanh(const DataType* data1, DataType* data2, int size){
	for (int i = 0; i < size; ++i){
		DataType e_pos = exp(data1[i]);
		DataType e_neg = exp(-data1[i]);
		data2[i] = (e_pos - e_neg) / (e_pos + e_neg);
	}
}

Tensor TensorOperation::sub(const Tensor& t1, const Tensor& t2, DataType* data){
	const Tensor* ptr1;
	const Tensor* ptr2;
	if (t1.get_shape().size() >= t2.get_shape().size()){
		ptr1 = &t1;
		ptr2 = &t2;
	}else{
		Logger::logging("tensor sub shape error!", "ERROR");
		return Tensor();
	}
	const std::vector<int>& shape1 = ptr1->get_shape();
	const std::vector<int>& shape2 = ptr2->get_shape();
	const DataType* data1 = ptr1->get_data();
	const DataType* data2 = ptr2->get_data();
	
	int size = 1;
	for (int i = 1; i <= shape2.size(); ++i){
		if (shape2[shape2.size() - i] != shape1[shape1.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return Tensor();
		}
		size *= shape2[shape2.size() - i]; 
	}

	int iters = ptr1->get_size() / size;

	Tensor ret(shape1, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}

	DataType* ret_data = ret.get_var_data();	

	if (size < 16){
		for (int i = 0; i < iters; ++i){
			for (int j = 0; j < size; ++j){
				ret_data[i * size + j] = data1[i * size + j] - data2[j];
			}
		}
	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_sub(data1 + (i *size) , data2, ret_data + (i * size), size); 
		}
	}
	return ret;
}

Tensor TensorOperation::div(const Tensor& t1, const Tensor& t2, DataType* data){
	const Tensor* ptr1;
	const Tensor* ptr2;
	if (t1.get_shape().size() >= t2.get_shape().size()){
		ptr1 = &t1;
		ptr2 = &t2;
	}else{
		Logger::logging("tensor div shape error!", "ERROR");
		return Tensor();
	}
	const std::vector<int>& shape1 = ptr1->get_shape();
	const std::vector<int>& shape2 = ptr2->get_shape();
	const DataType* data1 = ptr1->get_data();
	const DataType* data2 = ptr2->get_data();
	
	int size = 1;
	for (int i = 1; i <= shape2.size(); ++i){
		if (shape2[shape2.size() - i] != shape1[shape1.size() -i]){
			Logger::logging("tensor adding shape error !", "ERROR");
			return Tensor();
		}
		size *= shape2[shape2.size() - i]; 
	}

	int iters = ptr1->get_size() / size;

	Tensor ret(shape1, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}

	DataType* ret_data = ret.get_var_data();	

	if (size < 16){
		for (int i = 0; i < iters; ++i){
			for (int j = 0; j < size; ++j){
				ret_data[i * size + j] = data1[i * size + j] / data2[j];
			}
		}
	}else{
		for (int i = 0; i < iters; ++i){
			sse_vector_div(data1 + (i *size) , data2, ret_data + (i * size), size); 
		}
	}
	return ret;
}

Tensor TensorOperation::sqrt_t(const Tensor& t1, DataType* data){
	const std::vector<int>& shape1 = t1.get_shape();
	const DataType* data1 = t1.get_data();
	int size = t1.get_size();

	Tensor ret(shape1, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[size];
		ret.set_data(buff, true);
	}

	DataType* ret_data = ret.get_var_data();	

	if (size < 16){
		for (int i = 0; i < size; ++i){
			ret_data[i] = sqrt(data1[i]);
		}
	}else{
		sse_vector_sqrt(data1, ret_data, size); 
	}
	return ret;
}

Tensor TensorOperation::max(const Tensor& t1, int axis, DataType* data){
	if (!t1.get_size()){
		return Tensor();
	}
	const std::vector<int>& input_shape = t1.get_shape();
	if (axis >= input_shape.size()){
		Logger::logging("tensort maxing axis error!", "ERROR");
		return Tensor();
	}
	
	int outer_iters = 1;
	int inner_iters = 1;
	std::vector<int> output_shape;

	for (int i = 0; i < axis; ++i){
		outer_iters *= input_shape[i];
		output_shape.push_back(input_shape[i]);
	}
	for (int i = axis + 1; i < input_shape.size(); ++i){
		inner_iters *= input_shape[i];
		output_shape.push_back(input_shape[i]);
	}	
	int seq_len = input_shape[axis];
	int isize = inner_iters * seq_len;
	
	Tensor ret(output_shape, false);
	if (data){
		ret.set_data(data, false);
	}else{
		DataType* buff = new DataType[ret.get_size()];
		ret.set_data(buff, true);
	}
	DataType* ret_data = ret.get_var_data();
	const DataType* data1 = t1.get_data();

	int rindex = 0;
	for (int i = 0; i < outer_iters; ++i){
		for (int j = 0; j < inner_iters; ++j){
			int index = i * isize + j;
			DataType cur_max = DATA_TYPE_NEG_INF;
			for (int k = 0; k < seq_len; ++k){
				if (data1[index] > cur_max){
					cur_max = data1[index];
				}
				index += inner_iters;
			}
			ret_data[rindex++] = cur_max;
		}
	}
	
	return ret;
}
