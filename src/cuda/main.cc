
#include <iostream>
#include <memory>

#include "timer.h"
#include "tensor.h"
#include "cbsconv.h"


void print_tensor(const std::unique_ptr<Tensor> &tensor) {
  tensor->ToHost();
  const float *data = tensor->data();
  std::cout << "{ ";
  for (size_t i = 0; i < tensor->size() - 1; ++i) {
    std::cout << data[i] << ", ";
  }
  std::cout << data[tensor->size()];
  std::cout << " }" << std::endl;
}


// TODO(lromor): consider using non-blocking stream
// for asynchrnous hence non synchronized/faster calls.
int main(int argc, char *argv[]) {

  const unsigned kBatches = 2;
  const unsigned kInputChannels = 1;
  const unsigned kOutputChannels = 1;
  const unsigned kInputW = 16;
  const unsigned kInputH = 16;
  const unsigned kKernelW = 8;
  const unsigned kKernelH = 8;

  // We have 25 pixel kernel. Let's make 3 centers with size 3, 2, and 4 = 9
  // In one direction the kernel is:
  // x x x o o
  // o o o o o
  // o o x x o
  // x x x x o
  float kSeparableKernel[9] = {
    0.1f, 1.5f, 0.1f,
    0.7f, 0.7f,
    0.1f, 1.5f, 1.5f, 0.1f
  };
  const int ncenters = 3;
  unsigned int kSeparableKernelPos[ncenters * 2] = { 0, 0, 2, 2, 4, 0 };
  size_t kSeparableKernelSizes[ncenters] = { 3, 2, 4 };

  // Create the required tensors
  const TensorShape input_shape = { kBatches, kInputChannels, kInputH, kInputW };
  const TensorShape kernel_size = { kKernelH, kKernelW };
  const TensorShape weights_shape = { kOutputChannels, kInputChannels, ncenters };

  std::unique_ptr<Tensor> input_tensor = Tensor::Create(input_shape, Tensor::Storage::DEVICE);
  std::unique_ptr<Tensor> weights_tensor = Tensor::Create(weights_shape, Tensor::Storage::DEVICE);

  CardinalBSplineConvolution(
    input_tensor, kernel_size, weights_tensor, kSeparableKernel,
    kSeparableKernelPos, kSeparableKernelSizes);

  return 0;
}
