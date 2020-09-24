#ifndef _TIMER_H
#define _TIMER_H

#include <memory>
#include <cuda_runtime_api.h>


class CudaTimer {
public:
  static std::unique_ptr<CudaTimer> Create();
  bool Start();
  bool Stop();
  float GetElapsedTime();
  ~CudaTimer();
private:
  CudaTimer() {}
  bool Init();
  cudaEvent_t start_, stop_;
};




#endif // _TIMER_H
