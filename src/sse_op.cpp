#include <math.h>
#include <iostream>
#include "sse_op.h"
#include "logger.h"

void sse_vector_add(const DataType *f1, const DataType *f2, DataType *f3, int len){
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	const DataType* p1 = f1;
	const DataType* p2 = f2;
	DataType* p3 = f3;	
	__m128 m1, m2, m3;
	
	for (int i = 0; i < iters; ++i){
		m1 = _mm_loadu_ps(p1);
		m2 = _mm_loadu_ps(p2);
		m3 = _mm_add_ps(m1, m2);
		_mm_storeu_ps(p3, m3);
		p1 += size;
		p2 += size;
		p3 += size;
	}
	
	for (int i = size * iters; i < len; ++i){
		f3[i] = f1[i] + f2[i];
	}

}

void sse_vector_mul(const DataType *f1, const DataType *f2, DataType *f3, int len){
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	const DataType* p1 = f1;
	const DataType* p2 = f2;
	DataType* p3 = f3;	
	__m128 m1, m2, m3;

	for (int i = 0; i < iters; ++i){
		m1 = _mm_loadu_ps(p1);
		m2 = _mm_loadu_ps(p2);
		m3 = _mm_mul_ps(m1, m2);
		_mm_storeu_ps(p3, m3);
		p1 += size;
		p2 += size;
		p3 += size;
	}
	
	for (int i = size * iters; i < len; ++i){
		f3[i] = f1[i] * f2[i];
	}
}

void sse_vector_sum(const DataType *f, int len, DataType& value){
	value = 0.0;
	int size = 16 / sizeof(DataType);
	int iters = len / size;
	__m128 sum = _mm_setzero_ps();
	const __m128* m = (__m128*) f;
	for (int i = 0; i < iters; ++i, ++m){
		sum = _mm_add_ps(sum, *m);
	}
	DataType *dsum = (DataType *) &sum;
	for (int i = 0; i < size; ++i){
		value += dsum[i]; 
	}
	for (int i = iters * size; i < len; ++i){
		value += f[i];
	}
}

void sse_vector_sub(const DataType *f1, const DataType *f2, DataType *f3, int len){
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	const DataType* p1 = f1;
	const DataType* p2 = f2;
	DataType* p3 = f3;	
	__m128 m1, m2, m3;
	
	for (int i = 0; i < iters; ++i){
		m1 = _mm_loadu_ps(p1);
		m2 = _mm_loadu_ps(p2);
		m3 = _mm_sub_ps(m1, m2);
		_mm_storeu_ps(p3, m3);
		p1 += size;
		p2 += size;
		p3 += size;
	}

	for (int i = size * iters; i < len; ++i){
		f3[i] = f1[i] - f2[i];
	}
}

void sse_vector_div(const DataType *f1, const DataType *f2, DataType *f3, int len){
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	const DataType* p1 = f1;
	const DataType* p2 = f2;
	DataType* p3 = f3;	
	__m128 m1, m2, m3;
	
	for (int i = 0; i < iters; ++i){
		m1 = _mm_loadu_ps(p1);
		m2 = _mm_loadu_ps(p2);
		m3 = _mm_div_ps(m1, m2);
		_mm_storeu_ps(p3, m3);
		p1 += size;
		p2 += size;
		p3 += size;
	}

	for (int i = size * iters; i < len; ++i){
		f3[i] = f1[i] / f2[i];
	}
}

void sse_vector_sqrt(const DataType *f1, DataType *f2, int len){
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	const DataType* p1 = f1;
	DataType* p2 = f2;	
	__m128 m1, m2;
	
	for (int i = 0; i < iters; ++i){
		m1 = _mm_loadu_ps(p1);
		m2 = _mm_sqrt_ps(m1);
		_mm_storeu_ps(p2, m2);
		p1 += size;
		p2 += size;
	}

	for (int i = size * iters; i < len; ++i){
		f2[i] = sqrt(f1[i]);
	}
}

void sse_vector_dot_mul(const DataType *f1, const DataType *f2, int len, DataType& value){
	value = 0.0;
	int size = 16 / sizeof(DataType);
	int iters = len / size;

	__m128 sum = _mm_setzero_ps();
	const __m128* m1 = (__m128*) f1;
	const __m128* m2 = (__m128*) f2;
	for (int i = 0; i < iters; ++i, ++m1, ++m2){
		sum = _mm_add_ps(sum, _mm_mul_ps(*m1, *m2));
	}

	DataType *dsum = (DataType *) &sum;
	for (int i = 0; i < size; ++i){
		value += dsum[i];
	}
	for (int i = iters * size; i < len; ++i){
		value += f1[i] * f2[i];
	}
}
