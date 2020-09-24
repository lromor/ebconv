#include <algorithm>
#include <numeric>
#include <optional>
#include <cstring>
#include <new>
#include <cassert>
#include <iterator>
#include <functional>

#include "timer.h"

std::unique_ptr<CudaTimer> CudaTimer::Create() {
  std::unique_ptr<CudaTimer> timer(new CudaTimer());
  if (!timer->Init())
    return NULL;
  return timer;
}

bool CudaTimer::Start() {
  if (cudaEventRecord(start_) != cudaSuccess) {
    fprintf(stderr, "Failed starting record event.\n");
    return false;
  }
  return true;
}

bool CudaTimer::Stop() {
  if (cudaEventRecord(stop_) != cudaSuccess) {
    fprintf(stderr, "Failed stopping record event.\n");
    return false;
  }
  if (cudaEventSynchronize(stop_) != cudaSuccess) {
    fprintf(stderr, "Failed synchronizing.\n");
    return false;
  }
  return true;
}

float CudaTimer::GetElapsedTime() {
  float milliseconds = 0;
  if (cudaEventElapsedTime(
        &milliseconds, start_, stop_) != cudaSuccess) {
    fprintf(stderr, "Error while evaluating elapsed time.\n");
    return 0;
  }
  return milliseconds;
}

CudaTimer::~CudaTimer() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

bool CudaTimer::Init() {
  if (cudaEventCreate(&start_) != cudaSuccess) {
    fprintf(stderr, "Failed cuda event creation.\n");
    return false;
  }
  if (cudaEventCreate(&stop_) != cudaSuccess) {
    fprintf(stderr, "Failed cuda event creation.\n");
    return false;
  }
  return true;
}

