#ifndef _SSE_OP_
#define _SSE_OP_
#include <nmmintrin.h>
#include "data_type.h"

void sse_vector_add(const DataType *f1, const DataType *f2, DataType *f3, int len);
void sse_vector_mul(const DataType *f1, const DataType *f2, DataType *f3, int len);
void sse_vector_sub(const DataType *f1, const DataType *f2, DataType *f3, int len);
void sse_vector_div(const DataType *f1, const DataType *f2, DataType *f3, int len);
void sse_vector_sqrt(const DataType *f1, DataType *f2, int len);
void sse_vector_sum(const DataType *f, int len, DataType& value);
void sse_vector_dot_mul(const DataType *f1, const DataType *f2, int len, DataType& value);

#endif
