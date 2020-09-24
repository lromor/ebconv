#ifndef _CBSCONV_H
#define _CBSCONV_H

#include <vector>
#include <cassert>
#include <cstdio>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "timer.h"
#include "tensor.h"

void CardinalBSplineConvolution(
  const UniqueTensor &input, const TensorShape kernel_size,
  const UniqueTensor &weight, float *sk_data,
  unsigned int *sk_pos, size_t *sk_size);


#endif // _CBSCONV_H
