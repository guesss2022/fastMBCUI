#ifndef KL_TRAITS_H
#define KL_TRAITS_H

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/complex.h>

namespace klchol {

// -------------------- PDE types --------------------------
enum PDE_TYPE
{
  POISSON_FLOAT=0  // float version of POISSON
};

template <enum PDE_TYPE pde>
struct pde_trait;

template <>
struct pde_trait<PDE_TYPE::POISSON_FLOAT>
{
  typedef float   scalar_t;
  typedef int32_t index_t;
  typedef float   real_t;
  static const int d = 1;
};

// ----------------------- type traits -------------------------

template <bool is_float, typename T>
struct get_real;

template <typename T>
struct get_real<true, T>
{
  __host__ __device__ __forceinline__
  static auto run(const T &x) { return x; }
};

template <typename T>
struct get_real<false, T>
{
  __host__ __device__ __forceinline__
  static auto run(const T &x) { return x.real(); }
};

template <typename T>
struct conjugation;

template <>
struct conjugation<double>
{
  __host__ __device__
  void operator()(double &x) {}
};

template <>
struct conjugation<float>
{
  __host__ __device__
  void operator()(float &x) {}
};

template <>
struct conjugation<thrust::complex<double>>
{
  __host__ __device__ __forceinline__
  void operator()(thrust::complex<double> &x)
  {
    x = thrust::conj(x);
  }
};

template <>
struct conjugation<thrust::complex<float>>
{
  __host__ __device__ __forceinline__
  void operator()(thrust::complex<float> &x)
  {
    x = thrust::conj(x);
  }
};

// -------------------- type traits ----------------------------

/* Scalars */
template <typename T>
struct cuda_data_trait;

template <>
struct cuda_data_trait<float>
{
  static const cudaDataType_t dataType = CUDA_R_32F;
};

template <>
struct cuda_data_trait<double>
{
  static const cudaDataType_t dataType = CUDA_R_64F;
};

template <>
struct cuda_data_trait<thrust::complex<float>>
{
  static const cudaDataType_t dataType = CUDA_C_32F;
};

template <>
struct cuda_data_trait<thrust::complex<double>>
{
  static const cudaDataType_t dataType = CUDA_C_64F;
};

/* Integers */
template <typename T>
struct cuda_index_trait;

template <>
struct cuda_index_trait<int32_t>
{
  static const cusparseIndexType_t indexType = CUSPARSE_INDEX_32I;
};

template <>
struct cuda_index_trait<int64_t>
{
  static const cusparseIndexType_t indexType = CUSPARSE_INDEX_64I;
};

/* Transpose options */
template <bool is_float>
struct cuda_adjoint_trait;

template <>
struct cuda_adjoint_trait<true>
{
  static const cusparseOperation_t opType = CUSPARSE_OPERATION_TRANSPOSE;
  static const cublasOperation_t blas_opType = CUBLAS_OP_T;
};

template <>
struct cuda_adjoint_trait<false>
{
  static const cusparseOperation_t opType = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  static const cublasOperation_t blas_opType = CUBLAS_OP_C;
};

}
#endif
