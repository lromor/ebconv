#include <algorithm>
#include <numeric>
#include <optional>
#include <cstring>
#include <new>
#include <cassert>
#include <iterator>
#include <functional>
#include <iostream>

#include "cbsconv.h"

static inline unsigned int TensorSize(const TensorShape &shape) {
  return std::accumulate(
    shape.begin(), shape.end(), 1,
    [&](unsigned int a, unsigned int b) { return a * b; });
}

std::unique_ptr<Tensor> Tensor::Create(const TensorShape &shape, Storage storage) {
  return Create(NULL, shape, storage);
}

std::unique_ptr<Tensor> Tensor::Create(const float *data, const TensorShape &shape, Storage storage) {
  std::unique_ptr<Tensor> tensor(new Tensor(shape));
  if (!tensor->Init(data, storage))
    return NULL;
  return tensor;
}

Tensor::~Tensor() {
  if (storage_ == Storage::HOST)
    delete data_;
  else
    cudaFree(data_);
}

TensorShape Tensor::GetShape() { return shape_; }
float *Tensor::data() { return data_; }
size_t Tensor::size() const { return TensorSize(shape_); }

bool Tensor::ToDevice() {
  if (storage_ == Storage::DEVICE)
    return true;

  assert(data_ && storage_ == Storage::HOST);
  const unsigned int tensor_size = TensorSize(shape_);

  // Allocate the space in device.
  float *device_data = NULL;
  if (cudaMalloc((void **)&device_data, sizeof(float) * tensor_size) != cudaSuccess) {
    fprintf(stderr, "Failed tensor allocation.\n");
    return false;
  }

  // Copy from host to device.
  if (cudaMemcpy(
        device_data, data_, sizeof(float) * tensor_size,
        cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed moving data from host to cuda device.\n");
    return false;
  }
  delete data_;
  data_ = device_data;
  return true;
}

bool Tensor::ToHost() {
  if (storage_ == Storage::HOST)
    return true;
  assert(data_ && storage_ == Storage::DEVICE);
  const unsigned int tensor_size = TensorSize(shape_);

  // Allocate space in host.
  float *host_data = NULL;
  if (!(host_data = new (std::nothrow) float[tensor_size])) {
    fprintf(stderr, "Could not allocate data in host.\n");
    return false;
  }

  // Copy from device to host.
  if (cudaMemcpy(
        host_data, data_, sizeof(float) * tensor_size,
        cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed moving data from cuda device to host.\n");
    return false;
  }
  cudaFree(data_);
  // Update with new pointer
  data_ = host_data;
  return true;
}

bool Tensor::Init(const float *data, const Storage storage) {
  const unsigned int tensor_size = TensorSize(shape_);

  if (storage == Storage::HOST) {
    // Allocate data
    if (!(data_ = new (std::nothrow) float[tensor_size])) {
      fprintf(stderr, "Could not allocate data in host.\n");
      return false;
    }
    // Copy data if data is not NULL
    if (data)
      std::memcpy(data_, data, tensor_size * sizeof(float));
  } else {
    assert(storage == Storage::DEVICE);

    // Allocate the tensor
    if (auto ret = cudaMalloc((void **)&data_, sizeof(float) * tensor_size) != cudaSuccess) {
      fprintf(stderr, "Failed tensor allocation %d.\n", ret);
      return false;
    }

    if (data && cudaMemcpy(
          data_, data, sizeof(float) * tensor_size,
          cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Failed moving data to cuda device.\n");
      return false;
    }
  }
  storage_ = storage;
  return true;
}
