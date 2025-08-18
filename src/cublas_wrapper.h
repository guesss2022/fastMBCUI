#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <thrust/complex.h>

namespace klchol {

/// ---------------------- dot product ---------------------------
template <typename scalar_t>
struct DOTP;

template <>
struct DOTP<float>
{
  __forceinline__
  static void run(cublasHandle_t handle,
                  int n,
                  const void *x,
                  cudaDataType xType,
                  int incx,
                  const void *y,
                  cudaDataType yType,
                  int incy,
                  void *result,
                  cudaDataType resultType,
                  cudaDataType executionType)
  {
    CHECK_CUBLAS(cublasDotEx(
        handle,
        n,
        x,
        xType,
        incx,
        y,
        yType,
        incy,
        result,
        resultType,
        executionType));
  }  
};

// template <>
// struct DOTP<thrust::complex<float>>
// {
//   __forceinline__
//   static void run(cublasHandle_t handle,
//                   int n,
//                   const void *x,
//                   cudaDataType xType,
//                   int incx,
//                   const void *y,
//                   cudaDataType yType,
//                   int incy,
//                   void *result,
//                   cudaDataType resultType,
//                   cudaDataType executionType)
//   {
//     CHECK_CUBLAS(cublasDotcEx(
//         handle,
//         n,
//         x,
//         xType,
//         incx,
//         y,
//         yType,
//         incy,
//         result,
//         resultType,
//         executionType));
//   }
// };

/// ------------------- matrix-vector product --------------------
template <typename scalar_t>
struct GEMV;

template <>
struct GEMV<float>
{
  __forceinline__
  static void run(cublasHandle_t handle, cublasOperation_t trans,
                  int m, int n,
                  const float          *alpha,
                  const float          *A, int lda,
                  const float          *x, int incx,
                  const float          *beta,
                  float          *y, int incy)
  {
    CHECK_CUBLAS(cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
  }
};

// template <>
// struct GEMV<thrust::complex<float>>
// {
//   __forceinline__
//   static void run(cublasHandle_t handle, cublasOperation_t trans,
//                   int m, int n,
//                   const thrust::complex<float> *alpha,
//                   const thrust::complex<float> *A, int lda,
//                   const thrust::complex<float> *x, int incx,
//                   const thrust::complex<float> *beta,
//                   thrust::complex<float> *y, int incy)
//   {
//     CHECK_CUBLAS(cublasCgemv(handle, trans, m, n,
//                              reinterpret_cast<const cuComplex*>(alpha),
//                              reinterpret_cast<const cuComplex*>(A), lda,
//                              reinterpret_cast<const cuComplex*>(x), incx,
//                              reinterpret_cast<const cuComplex*>(beta),
//                              reinterpret_cast<cuComplex*>(y), incy));
//   }
// };

}
#endif
