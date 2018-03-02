#ifndef _TENSOR_OP_H_
#define _TENSOR_OP_H_

#include "tensor.h"


class TensorOperation{

public:
	static Tensor add(const Tensor& t1, const Tensor& t2, DataType* data = NULL);
	static Tensor mul(const Tensor& t1, const Tensor& t2, DataType* data = NULL);
	static Tensor sub(const Tensor& t1, const Tensor& t2, DataType* data = NULL);
	static Tensor div(const Tensor& t1, const Tensor& t2, DataType* data = NULL);
	static Tensor sqrt_t(const Tensor& t1, DataType* data = NULL);
	static Tensor matmul(const Tensor& t1, const Tensor& t2, DataType* data = NULL);
	static Tensor concat(const Tensor& t1, const Tensor& t2, int axis, DataType* data = NULL);
	static Tensor activation_func(const Tensor& t1, const std::string& type, DataType* data = NULL);
	static void activation_func_in_situ(Tensor& t1, const std::string& type);
	static void sigmoid(const DataType* data1, DataType* data2, int size);
	static void relu(const DataType* data1, DataType* data2, int size);
	static void tanh(const DataType* data1, DataType* data2, int size);
	static Tensor max(const Tensor& t1, int aixs, DataType* data = NULL);
};

#endif
