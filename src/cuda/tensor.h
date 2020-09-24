#ifndef _TENSOR_H
#define _TENSOR_H

#include <vector>
#include <memory>

typedef std::vector<size_t> TensorShape;

// TODO(lromor): Use pimpl?
class Tensor {
public:
  enum class Storage { HOST, DEVICE };

  static std::unique_ptr<Tensor> Create(
    const TensorShape &shape, Storage storage = Storage::HOST);

  static std::unique_ptr<Tensor> Create(
    const float *data, const TensorShape &shape, Storage storage = Storage::HOST);

  ~Tensor();

  TensorShape GetShape();
  float *data();
  size_t size() const;

  bool ToDevice();
  bool ToHost();

private:
  Tensor(TensorShape shape) : shape_(shape), data_(NULL) {}
  bool Init(const float *data, const Storage storage);
  Storage storage_;
  TensorShape shape_;
  float *data_;
};

typedef std::unique_ptr<Tensor> UniqueTensor;


#endif // _TENSOR_H
