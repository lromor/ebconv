#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <numeric>
#include <optional>
#include <cstring>
#include <new>
#include <cassert>
#include <iterator>
#include <functional>

#include "timer.h"
#include "tensor.h"
#include "cbsconv.h"

// Ideally we want to find the right
// partition of a slice of the input that minimizes
// long fetches from the global memory and maximize cache
// exploitation and shared data.
#define THREADS_PER_BLOCK 1
#define MAX_SEPARABLE_FILTERS_SIZE (1 << 12)
#define MAX_CENTERS 256
#define MAX_DIMS 8

// Macro to convert index layout
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

static inline bool IsPowerOfTwo(unsigned x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

// Hold the data in the device in some structured way.
typedef struct {
  float *data;
  size_t *shape;
  size_t ndim;
} DeviceTensor;

typedef struct {
  size_t batch_size;
  size_t input_channels;
  size_t output_channels;
  size_t ndims;
  size_t input_spatial_dims[MAX_DIMS]; // Up to 8 dimensions
  size_t kernel_spatial_dims[MAX_DIMS]; // Up to 8 dimensions
} Dimensions;


__constant__ Dimensions dims;
__constant__ float sp_data[MAX_SEPARABLE_FILTERS_SIZE];
__constant__ size_t sp_size[MAX_CENTERS];
__constant__ size_t sp_pos[MAX_DIMS * MAX_CENTERS];



__global__ void cbsconv(const DeviceTensor *input, const DeviceTensor *weights,
                        float *sk_data, unsigned int *sk_pos, size_t *sk_size, DeviceTensor *output) {
  //int blockBatch = blockIdx.x;

}

static TensorShape convolution_output_shape(TensorShape input_shape, TensorShape weight_shape) {
  TensorShape output_shape = {input_shape[0], weight_shape[0]};
  for (unsigned i = 2; i < input_shape.size(); ++i) {
    const unsigned int idim = input_shape[i];
    const unsigned int kdim = weight_shape[i];
    output_shape.push_back(idim - kdim + 1);
  }
  return output_shape;
}

void CardinalBSplineConvolution(
  const UniqueTensor &input, const TensorShape kernel_size,
  const UniqueTensor &weight, float *sk_data,
  unsigned int *sk_pos, size_t *sk_size) {

  // Check the kernel size is a pow of 2.
  // It just necessary for the prototype.
  for (const auto &s : kernel_size) {
    assert(IsPowerOfTwo(s));
  }

  // Move tensors to device.
  input->ToDevice();
  weight->ToDevice();

  Dimensions dims = {0};
  const auto input_shape = input->GetShape();
  const auto weight_shape = weight->GetShape();
  dims.batch_size = input_shape[0];
  dims.input_channels = input_shape[1];
  assert(dims.input_channels == weight_shape[1]);
  dims.output_channels = weight_shape[0];
  dims.ndims = input_shape.size() - 2;

  TensorShape virtual_weights_shape = { dims.output_channels, dims.input_channels };
  for (size_t i = 0; i < dims.ndims; ++i) {
    dims.input_spatial_dims[i] = input_shape[i + 2];
    const size_t ks = kernel_size[i];
    dims.kernel_spatial_dims[i] = ks;
    virtual_weights_shape.push_back(ks);
  }

  const TensorShape output_shape = convolution_output_shape(input_shape, virtual_weights_shape);
  std::unique_ptr<Tensor> weights_tensor = Tensor::Create(output_shape, Tensor::Storage::DEVICE);

  // We want to make a grid of blocks that splits the input
  // by the number of batch sizes (x of the grid) and the number of blocks
  // that will work on a single slice (y of the grid).
  // Ideally, a block will contain a set of threads each will take care of computing
  // a line of the sub-output. For the best performance, the elaborated input shared across the threads should fit
  // inside the shared_memory.
  //   dim3 dimBlock(batch_size, BLOCK_SIZE);
  // dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  // MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  //   dim3 dim
  //   cbsconv<<<batch_size, THREADS_PER_BLOCK>>>(input->data(), )
}
