#ifndef MACRO_H
#define MACRO_H

#include <iostream>

#define ASSERT(x)                                       \
  do {                                                  \
    if (!(x)) {                                         \
      std::cerr << "# error: assertion failed at\n";    \
      std::cerr << __FILE__ << " " << __LINE__ << "\n"; \
      std::cerr << "# for: " << #x << std::endl;        \
      exit(0);                                          \
    }                                                   \
  } while(0);

#define CHECK_CUDA(func)                                                \
  {                                                                     \
    cudaError_t status = (func);                                        \
    if (status != cudaSuccess) {                                        \
      printf("CUDA API failed at line %d with error: %s (%d)\n",        \
             __LINE__, cudaGetErrorString(status), status);             \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define CHECK_CUSPARSE(func)                                            \
  {                                                                     \
    cusparseStatus_t status = (func);                                   \
    if (status != CUSPARSE_STATUS_SUCCESS) {                            \
      printf("cuSPARSE API failed at line %d with error: %s (%d)\n",    \
             __LINE__, cusparseGetErrorString(status), status);         \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define CHECK_CUBLAS(func)                                      \
  {                                                             \
    cublasStatus_t status = (func);                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
      printf("CUBLAS API failed at line %d with error: %d\n",   \
             __LINE__, status);                                 \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  }

#define CHECK_CUSOLVER(err)                                             \
  {                                                                     \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                              \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#endif
