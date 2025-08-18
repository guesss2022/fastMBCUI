// Copyright (C) 2021 Jiong Chen

#include "kl_chol.h"
#include <iostream>
#include <Eigen/Sparse>
#include <thrust/reverse.h>
#include <thrust/binary_search.h>
#include "cublas_wrapper.h"

namespace klchol {

///-------------------- utility for local Cholesky ---------------------------
template <typename scalar_t, typename index_t>
struct dense_chol_fac;

template <typename index_t>
struct dense_chol_fac<float, index_t>
{
  __device__
  static void run(const index_t n, float * __restrict__ A)
  {
    // A = U^tU
    float dot_sum = 0;
    index_t j = 0, i = 0;
    for (j = 0; j < n; ++j) {
      // Uij
      for (i = 0; i < j; ++i) {
        dot_sum = thrust::inner_product(thrust::device, A+j*n, A+j*n+i, A+i*n, 0.0f);
        A[i+j*n] = (A[i+n*j]-dot_sum)/A[i+i*n];
      }
      // Ujj
      dot_sum = thrust::inner_product(thrust::device, A+j*n, A+j*n+i, A+i*n, 0.0f);
      A[i+j*n] = sqrtf(A[j*n+j]-dot_sum);
    }
  }
};

template <typename scalar_t, typename index_t>
__device__ __host__
void upper_tri_solve(const index_t                nU,
                     const scalar_t* __restrict__ U,
                     const index_t                nb,
                     scalar_t*       __restrict__ b)
{
  // x = U^{-1}b
  const index_t end = nb-1;
  for (index_t i = 0; i < nb; ++i) {
    for (index_t j = 0; j < i; ++j) {
      b[end-i] -= b[end-j]*U[end-i+(end-j)*nU];
    }
    b[end-i] /= U[end-i+(end-i)*nU];
  }
}

///-------------------- assemble least-square covariance ------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void ker_least_square_lhs(const index_t               n_src,
                                     const real_t * __restrict__ d_src_xyz,
                                     const real_t * __restrict__ d_src_nxyz,
                                     const index_t               n_bnd,
                                     const real_t * __restrict__ d_bnd_xyz,
                                     const real_t * __restrict__ d_bnd_nxyz,
                                     const real_t * __restrict__ d_param,
                                     scalar_t     * __restrict__ d_K_val,
                                     const int kf_id)
{
  const index_t iter = blockIdx.x*blockDim.x + threadIdx.x;
  const index_t nnz = n_bnd*n_src;
  if ( iter >= nnz ) {
    return;
  }
  
  const index_t col = iter/n_bnd, row = iter%n_bnd;
  const int d = pde_trait<pde>::d;
  const index_t stride = d*n_bnd;
  
  Eigen::Matrix<scalar_t, d, d> G;
  gf_summary<pde, scalar_t, real_t>::run(
      kf_id,
      &d_bnd_xyz[3*row],  &d_src_xyz[3*col],
      &d_bnd_nxyz[3*row], &d_src_nxyz[3*col],
      d_param, G.data());

  #pragma unroll (pde_trait<pde>::d*pde_trait<pde>::d)
  for (index_t i = 0; i < d*d; ++i) {
    const index_t c = i/d, r = i%d;
    d_K_val[stride*(d*col+c)+d*row+r] = *(G.data()+i);
  }  
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::build_cov_mat_LS(const KERNEL_TYPE ker_type,
                                          const real_t *h_aux,
                                          const index_t length)
{
  ASSERT(d_bnd_xyz_ && d_src_xyz_); // boundary and source points must be present for LS solve

  const int d = pde_trait<pde>::d;
  const index_t nnz = n_bnd_pts_*n_src_pts_;
  const real_t Gb = 1.0*(d*d)*nnz*sizeof(scalar_t)/1000/1000/1000;
  std::cout << "# number of source points=" << n_src_pts_ << std::endl;
  std::cout << "# mem for kernel mat=" << Gb << " Gb" << std::endl;

  // rows and cols of the constraint matrix
  K_rows_ = d*n_bnd_pts_;
  K_cols_ = d*n_src_pts_;
          
  if ( d_K_val_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_K_val_, K_rows_*K_cols_*sizeof(scalar_t)));
    CHECK_CUDA(cudaMemset(d_K_val_, 0, K_rows_*K_cols_*sizeof(scalar_t)));
  }

  ASSERT(length <= AUX_SIZE_);  
  if ( d_aux_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_aux_, AUX_SIZE_*sizeof(real_t)));
  }
  CHECK_CUDA(cudaMemcpy(d_aux_, h_aux, length*sizeof(real_t), cudaMemcpyHostToDevice));
  
  const index_t threads_num = 256;
  const index_t blocks_num = (nnz+threads_num-1)/threads_num;
  ker_least_square_lhs<pde, scalar_t, index_t, real_t>
      <<< blocks_num, threads_num >>>
      (n_src_pts_, d_src_xyz_, d_src_nxyz_,
       n_bnd_pts_, d_bnd_xyz_, d_bnd_nxyz_,
       d_aux_, d_K_val_, static_cast<index_t>(ker_type));
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::K(scalar_t *Kval) const
{
  CHECK_CUDA(cudaMemcpy(Kval, d_K_val_, K_rows_*K_cols_*sizeof(scalar_t), cudaMemcpyDeviceToHost));  
}

template <enum PDE_TYPE pde>
void cov_assembler<pde>::build_cov_rhs_LS(const scalar_t *h_rhs_bnd,
                                          const index_t   size_bnd,
                                          scalar_t       *h_rhs_src,
                                          const index_t   size_src)
{  
  const int d = pde_trait<pde>::d;
  ASSERT(size_bnd == d*n_bnd_pts_ && size_src == d*n_src_pts_);

  if ( d_rhs_bnd_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_rhs_bnd_, size_bnd*sizeof(scalar_t)));
  }
  if ( d_rhs_src_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_rhs_src_, size_src*sizeof(scalar_t)));
  }

  // copy rhs from boundary and evaluate  
  const scalar_t ALPHA = 1, BETA = 0; 
  CHECK_CUDA(cudaMemcpy(d_rhs_bnd_, h_rhs_bnd, size_bnd*sizeof(scalar_t), cudaMemcpyHostToDevice));
  GEMV<scalar_t>::run(
      handle_,
      cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
      K_rows_, K_cols_,
      &ALPHA,
      d_K_val_, K_rows_,
      d_rhs_bnd_, 1,
      &BETA,
      d_rhs_src_, 1);
  CHECK_CUDA(cudaMemcpy(h_rhs_src, d_rhs_src_, size_src*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

///------------------- supernodal ------------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void klchol_super_fac_asm_scalar(const index_t  n_super,
                                            const index_t * __restrict__ d_super_ptr,
                                            const index_t * __restrict__ d_super_ind,
                                            const index_t * __restrict__ d_ptr,
                                            const index_t * __restrict__ d_ind,
                                            scalar_t      * __restrict__ d_val,
                                            const real_t  * __restrict__ d_xyz,
                                            const real_t  * __restrict__ d_nxyz,
                                            const real_t  * __restrict__ d_param,
                                            const index_t * __restrict__ d_TH_ptr,
                                            scalar_t      * __restrict__ d_TH_val,
                                            const int kf_id,
                                            const index_t                ROWS_COV,
                                            const scalar_t *__restrict__ d_LS_COV)
{
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }

  const auto TH_iter = d_TH_ptr[sid];
  const auto first_dof = d_super_ind[d_super_ptr[sid]];
  const index_t local_n = d_ptr[first_dof+1]-d_ptr[first_dof];

  if ( d_LS_COV == NULL ) {
    for (auto iter_i = d_ptr[first_dof], p = 0; iter_i < d_ptr[first_dof+1]; ++iter_i, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[first_dof+1]; ++iter_j, ++q) {
        const auto I = d_ind[iter_i], J = d_ind[iter_j];
        scalar_t G = 0;
        gf_summary<pde, scalar_t, real_t>::run(
            kf_id, &d_xyz[3*I], &d_xyz[3*J], &d_nxyz[3*I], &d_nxyz[3*J],
            d_param, &G);

        // reverse ordering for THETA
        const index_t rev_p = local_n-1-p, rev_q = local_n-1-q;
        d_TH_val[TH_iter+rev_p+rev_q*local_n] = G;
        conjugation<scalar_t>()(G);
        d_TH_val[TH_iter+rev_q+rev_p*local_n] = G;
      }
    }
  } else {
    for (auto iter_i = d_ptr[first_dof], p = 0; iter_i < d_ptr[first_dof+1]; ++iter_i, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[first_dof+1]; ++iter_j, ++q) {
        const auto I = d_ind[iter_i], J = d_ind[iter_j];
        scalar_t G = 0;
        gf_summary<pde, scalar_t, real_t>::run(I, J, ROWS_COV/pde_trait<pde>::d, d_LS_COV, &G);

        // reverse ordering for THETA
        const index_t rev_p = local_n-1-p, rev_q = local_n-1-q;
        d_TH_val[TH_iter+rev_p+rev_q*local_n] = G;
        conjugation<scalar_t>()(G);
        d_TH_val[TH_iter+rev_q+rev_p*local_n] = G;
      }
    }
  }
}

template <enum PDE_TYPE pde, typename scalar_t, typename index_t, typename real_t>
__global__ void klchol_super_fac_asm_vector(const index_t  n_super,
                                            const index_t * __restrict__ d_super_ptr,
                                            const index_t * __restrict__ d_super_ind,
                                            const index_t * __restrict__ d_ptr,
                                            const index_t * __restrict__ d_ind,
                                            scalar_t      * __restrict__ d_val,
                                            const real_t  * __restrict__ d_xyz,
                                            const real_t  * __restrict__ d_nxyz,
                                            const real_t  * __restrict__ d_param,
                                            const index_t * __restrict__ d_TH_ptr,
                                            scalar_t      * __restrict__ d_TH_val,
                                            const int kf_id,
                                            const index_t                ROWS_COV,
                                            const scalar_t *__restrict__ d_LS_COV)
{
  typedef Eigen::Matrix<scalar_t, 3, 3> Mat3f;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vec3f;  
  
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }

  const auto TH_iter = d_TH_ptr[sid];
  const auto first_dof = d_super_ind[d_super_ptr[sid]];
  const index_t local_n = d_ptr[3*first_dof+1]-d_ptr[3*first_dof];

  if ( d_LS_COV == NULL ) {
    for (auto iter_i = d_ptr[3*first_dof], p = 0; iter_i < d_ptr[3*first_dof+1]; iter_i += 3, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[3*first_dof+1]; iter_j += 3, ++q) {
        const auto I = d_ind[iter_i]/3, J = d_ind[iter_j]/3;

        Mat3f G = Mat3f::Zero();
        gf_summary<pde, scalar_t, real_t>::run(
            kf_id, &d_xyz[3*I], &d_xyz[3*J], &d_nxyz[3*I], &d_nxyz[3*J],
            d_param, G.data());

        // reverse ordering for THETA
        index_t rev_p, rev_q;

        rev_p = local_n/3-1-p; rev_q = local_n/3-1-q;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(0, 1);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(1, 0);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(0, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(2, 0);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(1, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(2, 1);
        }

        rev_p = local_n/3-1-q; rev_q = local_n/3-1-p;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(1, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(0, 1);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(2, 0);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(0, 2);
          
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(2, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(1, 2);
        }
      }
    }
  } else {
    for (auto iter_i = d_ptr[3*first_dof], p = 0; iter_i < d_ptr[3*first_dof+1]; iter_i += 3, ++p) {
      for (auto iter_j = iter_i, q = p; iter_j < d_ptr[3*first_dof+1]; iter_j += 3, ++q) {
        const auto I = d_ind[iter_i]/3, J = d_ind[iter_j]/3;

        Mat3f G = Mat3f::Zero();
        gf_summary<pde, scalar_t, real_t>::run(I, J, ROWS_COV/pde_trait<pde>::d, d_LS_COV, G.data());
        
        // reverse ordering for THETA, G could be unsymmetric!!!
        index_t rev_p, rev_q;
        
        rev_p = local_n/3-1-p; rev_q = local_n/3-1-q;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(0, 1);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(1, 0);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(0, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(2, 0);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(1, 2);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(2, 1);
        }

        rev_p = local_n/3-1-q; rev_q = local_n/3-1-p;
        {
          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+2)*local_n] = G(0, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+1)*local_n] = G(1, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+0)*local_n] = G(2, 2);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+1)*local_n] = G(1, 0);
          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+2)*local_n] = G(0, 1);

          d_TH_val[TH_iter+(3*rev_p+2)+(3*rev_q+0)*local_n] = G(2, 0);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+2)*local_n] = G(0, 2);

          d_TH_val[TH_iter+(3*rev_p+1)+(3*rev_q+0)*local_n] = G(2, 1);
          d_TH_val[TH_iter+(3*rev_p+0)+(3*rev_q+1)*local_n] = G(1, 2);
        }
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void klchol_super_fac_chol(const index_t n_super,
                                      const index_t * __restrict__ d_TH_ptr,
                                      scalar_t      * __restrict__ d_TH_val)
{
  // for each supernode
  const index_t sid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( sid >= n_super ) {
    return;
  }
  const index_t local_n = floor(sqrt((double)(d_TH_ptr[sid+1]-d_TH_ptr[sid]))+0.5);
  const auto TH_iter = d_TH_ptr[sid];
  dense_chol_fac<scalar_t, index_t>::run(local_n, &d_TH_val[TH_iter]);
}

template <typename scalar_t, typename index_t>
__global__ void klchol_super_fac_bs(const index_t n,
                                    const index_t dim,
                                    const index_t * __restrict__  d_ptr,
                                    const index_t * __restrict__  d_ind,
                                    scalar_t      * __restrict__  d_val,
                                    const index_t * __restrict__  d_super_parent,
                                    const index_t * __restrict__  d_TH_ptr,
                                    const scalar_t * __restrict__ d_TH_val)
{
  const index_t pid = blockIdx.x*blockDim.x + threadIdx.x;
  if ( pid >= n ) {
    return;
  }

  const index_t node_id = pid/dim;
  const index_t sid = d_super_parent[node_id];
  const index_t nU = floor(sqrt((double)(d_TH_ptr[sid+1]-d_TH_ptr[sid]))+0.5);
  const index_t nb = d_ptr[pid+1]-d_ptr[pid]; 
  const auto ptr_pid = d_ptr[pid];

  d_val[d_ptr[pid+1]-1] = 1.0;
  upper_tri_solve(nU, &d_TH_val[d_TH_ptr[sid]], nb, &d_val[ptr_pid]);
  thrust::reverse(thrust::device, d_val+ptr_pid, d_val+ptr_pid+nb);
  thrust::for_each(thrust::device, d_val+ptr_pid, d_val+ptr_pid+nb, conjugation<scalar_t>());
}

//==============================================================
template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::gpu_simpl_klchol(const index_t npts, const index_t ker_dim, const index_t num_sec)
    : npts_(npts), ker_dim_(ker_dim), n_(npts*ker_dim), num_sec_(num_sec)
{
  ASSERT(num_sec_ <= npts_);

  // init cusparse and cusolverdn handles
  CHECK_CUSPARSE(cusparseCreate(&cus_handle_));
  CHECK_CUBLAS(cublasCreate(&bls_handle_));
  
  // malloc b and x: Nx1 buffer
  CHECK_CUDA(cudaMalloc((void **)&d_vecB_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_vecX_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemset(d_vecB_, 0, n_*sizeof(scalar_t)));  
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_*sizeof(scalar_t)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&b_, n_, d_vecB_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&x_, n_, d_vecX_, cuda_data_trait<scalar_t>::dataType));

  //  // triangular solve
  //  CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv_desc_));

  // malloc buffer for *partitioned* evaluation
  CHECK_CUDA(cudaMalloc((void **)&d_sec_eval_, n_*num_sec_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMemset(d_sec_eval_, 0, n_*num_sec_*sizeof(scalar_t)));

  // vectors for pcg
  CHECK_CUDA(cudaMalloc((void **)&d_resd_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_temp_, n_*sizeof(scalar_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_z_,    n_*sizeof(scalar_t)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&resd_, n_, d_resd_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&temp_, n_, d_temp_, cuda_data_trait<scalar_t>::dataType));
  CHECK_CUSPARSE(cusparseCreateDnVec(&z_, n_, d_z_, cuda_data_trait<scalar_t>::dataType));

  // explicit covariance matrix
  ls_cov_ = new covariance_t();
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::~gpu_simpl_klchol()
{
  if ( cus_handle_ ) { CHECK_CUSPARSE(cusparseDestroy(cus_handle_)); }
  if ( bls_handle_ ) { CHECK_CUBLAS(cublasDestroy(bls_handle_)); }
  
  if ( d_ptr_) { CHECK_CUDA(cudaFree(d_ptr_)); }
  if ( d_ind_) { CHECK_CUDA(cudaFree(d_ind_)); }
  if ( d_val_) { CHECK_CUDA(cudaFree(d_val_)); }
  
  if ( d_vecB_) { CHECK_CUDA(cudaFree(d_vecB_)); }
  if ( d_vecX_) { CHECK_CUDA(cudaFree(d_vecX_)); }

  if ( d_resd_) { CHECK_CUDA(cudaFree(d_resd_)); }
  if ( d_temp_) { CHECK_CUDA(cudaFree(d_temp_)); }
  if ( d_z_   ) { CHECK_CUDA(cudaFree(d_z_));    }

  if ( d_ker_aux_ ) { CHECK_CUDA(cudaFree(d_ker_aux_)); }
  
  if ( d_xyz_ )  { CHECK_CUDA(cudaFree(d_xyz_));  }
  if ( d_nxyz_ ) { CHECK_CUDA(cudaFree(d_nxyz_)); }

  if ( d_pred_xyz_ ) { CHECK_CUDA(cudaFree(d_pred_xyz_)); }
  if ( d_pred_f_   ) { CHECK_CUDA(cudaFree(d_pred_f_));   }

  if ( d_TH_ptr_ ) { CHECK_CUDA(cudaFree(d_TH_ptr_)); }
  if ( d_TH_val_ ) { CHECK_CUDA(cudaFree(d_TH_val_)); }

  if ( d_work_) { CHECK_CUDA(cudaFree(d_work_)); }
  //  if ( d_sv_work_ ) { CHECK_CUDA(cudaFree(d_sv_work_)); }
  
  if ( A_ ) { CHECK_CUSPARSE(cusparseDestroySpMat(A_)); }
  if ( b_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(b_)); }
  if ( x_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(x_)); }

  //  if ( spsv_desc_ ) { CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsv_desc_)); }

  if ( resd_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(resd_)); }
  if ( temp_ ) { CHECK_CUSPARSE(cusparseDestroyDnVec(temp_)); }
  if ( z_    ) { CHECK_CUSPARSE(cusparseDestroyDnVec(z_));    }

  if ( ls_cov_ ) { delete ls_cov_; }
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_kernel(const KERNEL_TYPE ker_type,
                                       const real_t      *h_ker_aux,
                                       const index_t     length)
{
  ASSERT(length <= AUX_LENGTH_);

  if ( d_ker_aux_ == NULL ) {
    CHECK_CUDA(cudaMalloc((void **)&d_ker_aux_, AUX_LENGTH_*sizeof(real_t)));
  }
  ker_type_ = ker_type;
  CHECK_CUDA(cudaMemcpy(d_ker_aux_, h_ker_aux, length*sizeof(real_t), cudaMemcpyHostToDevice));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::set_sppatt(const index_t n,
                                  const index_t nnz,
                                  const index_t *h_ptr,
                                  const index_t *h_ind)
{
  ASSERT(n == n_);
  const auto &malloc_factor =
      [&]() {
        CHECK_CUDA(cudaMalloc((void **)&d_ptr_, (n_+1)*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_ind_, nnz_*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_val_, nnz_*sizeof(scalar_t)));
        CHECK_CUSPARSE(cusparseCreateCsr(&A_, n_, n_, nnz_, d_ptr_, d_ind_, d_val_,
                                         cuda_index_trait<index_t>::indexType,
                                         cuda_index_trait<index_t>::indexType,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_data_trait<scalar_t>::dataType));
      };
  
  const auto &malloc_theta = 
      [&]() {
        // for each spatial point...
        std::vector<index_t> ptr(npts_+1, 0);
        for (index_t j = 0; j < npts_; ++j) {
          const index_t nnz_j = h_ptr[ker_dim_*j+1]-h_ptr[ker_dim_*j];
          ASSERT(nnz_j%ker_dim_ == 0);
          ptr[j+1] = ptr[j]+nnz_j*nnz_j;
        }
        TH_nnz_ = ptr.back();
        const real_t Gb = TH_nnz_*sizeof(scalar_t)/(1024*1024*1024);
        // spdlog::info("total size for THETA={0:.1f}, {1:.2f}", TH_nnz_, Gb);
        ASSERT(TH_nnz_ < INT_MAX);
        
        CHECK_CUDA(cudaMalloc((void **)&d_TH_ptr_, ptr.size()*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&d_TH_val_, ptr.back()*sizeof(scalar_t)));
        CHECK_CUDA(cudaMemcpy(d_TH_ptr_, &ptr[0], ptr.size()*sizeof(index_t), cudaMemcpyHostToDevice));
      };

  const auto &malloc_mv_buffer =
      [&]() {
        const scalar_t alpha = 1, beta = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            cus_handle_,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A_, b_, &beta, x_,
            cuda_data_trait<scalar_t>::dataType,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &buff_sz_));    
        CHECK_CUDA(cudaMalloc(&d_work_, buff_sz_));
        // spdlog::info("SpMV buffer size={}", buff_sz_);
      };

  if ( d_ptr_ == NULL ) { // allocate for the first time
    nnz_ = nnz;

    malloc_factor();
    malloc_theta();
    malloc_mv_buffer();
  }

  if ( d_ptr_ && nnz_ != nnz ) { // reallocate if nnz is changed
    nnz_ = nnz;

    CHECK_CUDA(cudaFree(d_ptr_));
    CHECK_CUDA(cudaFree(d_ind_));
    CHECK_CUDA(cudaFree(d_val_));
    CHECK_CUSPARSE(cusparseDestroySpMat(A_));
    malloc_factor();

    CHECK_CUDA(cudaFree(d_TH_ptr_));
    CHECK_CUDA(cudaFree(d_TH_val_));
    malloc_theta();

    CHECK_CUDA(cudaFree(d_work_));
    malloc_mv_buffer();
  }

  // copy factor sparsity
  CHECK_CUDA(cudaMemcpy(d_ptr_, h_ptr, (n_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_ind_, h_ind, nnz_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_val_, 0, nnz_*sizeof(scalar_t)));
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::get_factor(Eigen::SparseMatrix<scalar_t> &L) const
{
  Eigen::Matrix<index_t, -1, 1> ptr(n_+1), ind(nnz_);
  Eigen::Matrix<scalar_t, -1, 1> val(nnz_);  

  CHECK_CUDA(cudaMemcpy(ptr.data(), d_ptr_, (n_+1)*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ind.data(), d_ind_, nnz_*sizeof(index_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(val.data(), d_val_, nnz_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  L = Eigen::Map<Eigen::SparseMatrix<scalar_t>>(n_, n_, nnz_, ptr.data(), ind.data(), val.data());
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::real_t gpu_simpl_klchol<pde>::memory() const {
  return 0;
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::compute()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::assemble()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::factorize()
{
  bool simplicial_factorization_is_deprecated = false;
  ASSERT(simplicial_factorization_is_deprecated);
}

template <enum PDE_TYPE pde>
void gpu_simpl_klchol<pde>::solve(const scalar_t *h_rhs, scalar_t *h_x)
{
  const scalar_t alpha = 1, beta = 0;

  CHECK_CUDA(cudaMemcpy(d_vecX_, h_rhs, n_*sizeof(scalar_t), cudaMemcpyHostToDevice));
 
  // Note that A_ is of csr format, so no transpose for x = L^T*b
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, A_, x_, &beta, b_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
  // then b = L*x
  CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                              cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                              &alpha, A_, b_, &beta, x_,
                              cuda_data_trait<scalar_t>::dataType,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_work_));

  CHECK_CUDA(cudaMemcpy(h_x, d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}


template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::gpu_super_klchol(const index_t npts, const index_t ker_dim, const index_t sec_num)
    : gpu_simpl_klchol<pde>(npts, ker_dim, sec_num)
{
}

template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::~gpu_super_klchol()
{
  if ( d_super_ptr_    ) { CHECK_CUDA(cudaFree(d_super_ptr_));    }
  if ( d_super_ind_    ) { CHECK_CUDA(cudaFree(d_super_ind_));    }
  if ( d_super_parent_ ) { CHECK_CUDA(cudaFree(d_super_parent_)); }

  if ( h_super_ptr_ ) { delete[] h_super_ptr_; }
  if ( h_super_ind_ ) { delete[] h_super_ind_; }
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::set_supernodes(const index_t n_super,
                                      const index_t n_pts,
                                      const index_t *h_super_ptr,
                                      const index_t *h_super_ind,
                                      const index_t *h_super_parent)
{
  ASSERT(this->npts_ == n_pts);
  ASSERT(*std::max_element(h_super_parent, h_super_parent+n_pts)+1 == n_super);

  n_super_ = n_super;
  // spdlog::info("n_super_={}, npts_={}", n_super_, this->npts_);

  // store a copy on cpu
  h_super_ptr_ = new index_t[n_super_+1];
  h_super_ind_ = new index_t[this->npts_];
  std::copy(h_super_ptr, h_super_ptr+n_super_+1, h_super_ptr_);
  std::copy(h_super_ind, h_super_ind+this->npts_, h_super_ind_);

  // gpu memory allocation
  CHECK_CUDA(cudaMalloc((void **)&d_super_ptr_,    (n_super_+1)*sizeof(index_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_super_ind_,    this->npts_*sizeof(index_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_super_parent_, this->npts_*sizeof(index_t)));

  CHECK_CUDA(cudaMemcpy(d_super_ptr_, h_super_ptr,    (n_super_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_super_ind_, h_super_ind,    this->npts_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_super_parent_, h_super_parent, this->npts_*sizeof(index_t), cudaMemcpyHostToDevice));
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::set_sppatt(const index_t n,
                                  const index_t nnz,
                                  const index_t *h_ptr,
                                  const index_t *h_ind)
{
  ASSERT(n == this->n_ && this->npts_*this->ker_dim_ == n);
  ASSERT(d_super_ptr_ != NULL && d_super_ind_ != NULL);

  const auto &malloc_factor =
      [&]() {
        CHECK_CUDA(cudaMalloc((void **)&this->d_ptr_, (this->n_+1)*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_ind_, this->nnz_*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_val_, this->nnz_*sizeof(scalar_t)));
        CHECK_CUSPARSE(cusparseCreateCsr(&this->A_, this->n_, this->n_, this->nnz_, this->d_ptr_, this->d_ind_, this->d_val_,
                                         cuda_index_trait<index_t>::indexType,
                                         cuda_index_trait<index_t>::indexType,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_data_trait<scalar_t>::dataType));
      };

  const auto &malloc_theta = 
      [&]() {
        std::vector<index_t> ptr(n_super_+1, 0);
        for (index_t j = 0; j < n_super_; ++j) {
          // for each super node
          const index_t super_iter = h_super_ptr_[j];
          const index_t first_dof = this->ker_dim_*h_super_ind_[super_iter];
          const index_t nnz_super_j = h_ptr[first_dof+1]-h_ptr[first_dof];
          ptr[j+1] = ptr[j]+nnz_super_j*nnz_super_j;
        }
        this->TH_nnz_ = ptr.back();
        const real_t Gb = this->TH_nnz_*sizeof(scalar_t)/(1024*1024*1024);
        // spdlog::info("total size for THETA={0:.1f}, {1:.2f} GB", this->TH_nnz_, Gb);
        ASSERT(this->TH_nnz_ < INT_MAX);        

        CHECK_CUDA(cudaMalloc((void **)&this->d_TH_ptr_, ptr.size()*sizeof(index_t)));
        CHECK_CUDA(cudaMalloc((void **)&this->d_TH_val_, ptr.back()*sizeof(scalar_t)));
        CHECK_CUDA(cudaMemcpy(this->d_TH_ptr_, &ptr[0], ptr.size()*sizeof(index_t), cudaMemcpyHostToDevice));
      };

  const auto &malloc_mv_buffer =
      [&]() {
        const scalar_t alpha = 1, beta = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            this->cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, this->A_, this->b_, &beta, this->x_,
            cuda_data_trait<scalar_t>::dataType,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &this->buff_sz_));    
        CHECK_CUDA(cudaMalloc(&this->d_work_, this->buff_sz_));
        // spdlog::info("SpMV buffer size={}", this->buff_sz_);
      };

  if ( this->d_ptr_ == NULL ) { // allocate for the first time
    this->nnz_ = nnz;

    malloc_factor();
    malloc_theta();
    malloc_mv_buffer();
  }

  if ( this->d_ptr_ && this->nnz_ != nnz ) { // reallocate if nnz is changed
    this->nnz_ = nnz;

    CHECK_CUDA(cudaFree(this->d_ptr_));
    CHECK_CUDA(cudaFree(this->d_ind_));
    CHECK_CUDA(cudaFree(this->d_val_));
    CHECK_CUSPARSE(cusparseDestroySpMat(this->A_));
    malloc_factor();

    CHECK_CUDA(cudaFree(this->d_TH_ptr_));
    CHECK_CUDA(cudaFree(this->d_TH_val_));
    malloc_theta();

    CHECK_CUDA(cudaFree(this->d_work_));
    malloc_mv_buffer();
  }

  // copy factor sparsity
  CHECK_CUDA(cudaMemcpy(this->d_ptr_, h_ptr, (this->n_+1)*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(this->d_ind_, h_ind, this->nnz_*sizeof(index_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));
}

template <enum PDE_TYPE pde>
gpu_super_klchol<pde>::real_t gpu_super_klchol<pde>::memory() const {
  const real_t total_bytes = 
      (this->n_+1)*sizeof(index_t)+
      this->nnz_*sizeof(index_t)+
      this->nnz_*sizeof(scalar_t)+
      (n_super_+1)*sizeof(index_t)+
      this->TH_nnz_*sizeof(scalar_t);  
  return total_bytes/1024/1024/1024;
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::compute()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));

  // assemble
  if ( this->ker_dim_ == 1 ) {
    klchol_super_fac_asm_scalar<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  } else if ( this->ker_dim_ == 3 ) {
    klchol_super_fac_asm_vector<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  }

  klchol_super_fac_chol<<< blocks_super, threads_num >>>
      (n_super_, this->d_TH_ptr_, this->d_TH_val_);

  klchol_super_fac_bs<<< blocks_dof, threads_num >>>
      (this->n_, this->ker_dim_, this->d_ptr_, this->d_ind_, this->d_val_,
       d_super_parent_, this->d_TH_ptr_, this->d_TH_val_);
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::assemble()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  CHECK_CUDA(cudaMemset(this->d_val_, 0, this->nnz_*sizeof(scalar_t)));

  if ( this->ker_dim_ == 1 ) {
    klchol_super_fac_asm_scalar<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  } else if ( this->ker_dim_ == 3 ) {
    klchol_super_fac_asm_vector<pde, scalar_t, index_t, real_t>
        <<< blocks_super, threads_num >>>
        (n_super_,
         d_super_ptr_, d_super_ind_,
         this->d_ptr_, this->d_ind_, this->d_val_,
         this->d_xyz_, this->d_nxyz_, this->d_ker_aux_,
         this->d_TH_ptr_, this->d_TH_val_,
         static_cast<int>(this->ker_type_),
         this->ls_cov_->K_rows_,
         this->ls_cov_->d_K_val_);
  }
}

template <enum PDE_TYPE pde>
void gpu_super_klchol<pde>::factorize()
{
  const index_t threads_num = 256;
  const index_t blocks_super = (n_super_+threads_num-1)/threads_num;
  const index_t blocks_dof   = (this->n_+threads_num-1)/threads_num;

  klchol_super_fac_chol<<< blocks_super, threads_num >>>
      (n_super_, this->d_TH_ptr_, this->d_TH_val_);
  
  klchol_super_fac_bs<<< blocks_dof, threads_num >>>
      (this->n_, this->ker_dim_, this->d_ptr_, this->d_ind_, this->d_val_,
       d_super_parent_, this->d_TH_ptr_, this->d_TH_val_);
}

template class gpu_simpl_klchol<PDE_TYPE::POISSON_FLOAT>;
template class gpu_super_klchol<PDE_TYPE::POISSON_FLOAT>;
template struct cov_assembler<PDE_TYPE::POISSON_FLOAT>;

template <enum PDE_TYPE pde>
void cov_assembler<pde>::build_cov_mat_LS_float(const float* M_p_data, const int K_rows, const int K_cols, 
                                                      const KERNEL_TYPE ktype, const float* param, const int param_size)
{
  // Store parameters to d_aux_
  if (d_aux_ == NULL) {
    CHECK_CUDA(cudaMalloc((void **)&d_aux_, AUX_SIZE_*sizeof(real_t)));
  }
  CHECK_CUDA(cudaMemcpy(d_aux_, param, param_size*sizeof(real_t), cudaMemcpyHostToDevice));
  
  // Set matrix sizes
  K_rows_ = K_rows;
  K_cols_ = K_cols;
  
  // Allocate memory for d_K_val_
  if (d_K_val_ != NULL) {
    CHECK_CUDA(cudaFree(d_K_val_));
  }
  CHECK_CUDA(cudaMalloc((void **)&d_K_val_, K_rows_*K_cols_*sizeof(scalar_t)));
  
  // Copy float data from M_p_data directly to d_K_val_
  CHECK_CUDA(cudaMemcpy(d_K_val_, M_p_data, K_rows_*K_cols_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::pcg_ret_t
gpu_simpl_klchol<pde>::pcg_guesss(const scalar_t *h_rhs_bnd,
                               const index_t size_bnd,
                               scalar_t *h_x,
                               const bool preconditioned,
                               const index_t maxits,
                               const real_t TOL,
                               const bool verbose,
                               std::vector<real_t> *residual)
{
  // Check parameters
  const int d = pde_trait<pde>::d;
  ASSERT(ls_cov_->d_K_val_ != NULL); // Must be a least-squares problem
  ASSERT(size_bnd == ls_cov_->K_rows_);

  const index_t threads_num = 256;
  const index_t blocks_num = (npts_+threads_num-1)/threads_num;
  const index_t sec_blocks_num = (npts_*num_sec_+threads_num-1)/threads_num;
  const scalar_t ALPHA = 1, BETA = 0;

  if (residual) {
    residual->clear();
    residual->reserve(2*maxits);
    residual->emplace_back(1.0);
  }

  // Copy boundary right-hand side directly to GPU
  if (ls_cov_->d_rhs_bnd_ == NULL) {
    CHECK_CUDA(cudaMalloc((void **)&ls_cov_->d_rhs_bnd_, size_bnd*sizeof(scalar_t)));
  }
  CHECK_CUDA(cudaMemcpy(ls_cov_->d_rhs_bnd_, h_rhs_bnd, size_bnd*sizeof(scalar_t), cudaMemcpyHostToDevice));

  // Compute right-hand side d_rhs_src_ on GPU
  if (ls_cov_->d_rhs_src_ == NULL) {
    CHECK_CUDA(cudaMalloc((void **)&ls_cov_->d_rhs_src_, n_*sizeof(scalar_t)));
  }
  GEMV<scalar_t>::run(
      ls_cov_->handle_,
      cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
      ls_cov_->K_rows_, ls_cov_->K_cols_,
      &ALPHA,
      ls_cov_->d_K_val_, ls_cov_->K_rows_,
      ls_cov_->d_rhs_bnd_, 1,
      &BETA,
      ls_cov_->d_rhs_src_, 1);

  // Assign rhs_src directly to d_resd_ without going through CPU
  CHECK_CUDA(cudaMemcpy(d_resd_, ls_cov_->d_rhs_src_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_*sizeof(scalar_t)));

  scalar_t RHS2 = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      &RHS2,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);
  if (get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2) < 1e-16) {
    CHECK_CUDA(cudaMemset(h_x, 0, n_*sizeof(scalar_t)));
    return std::make_pair(index_t(0), real_t(0));
  }

  const real_t threshold = TOL*TOL*get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2);
  if(verbose){
    std::cout << "pcg debug!!!!!!!!!!\nthreshold: " << threshold << std::endl;
  }

  // solve A*b = resd
  if (preconditioned) {
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ALPHA, A_, resd_, &BETA, temp_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                &ALPHA, A_, temp_, &BETA, b_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));    
  } else {
    CHECK_CUDA(cudaMemcpy(d_vecB_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
  }

  scalar_t absNew = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
      &absNew,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);

  index_t i = 0;
  scalar_t RESD2 = 0;

  while (i < maxits) {
    ++i;
    
    // Compute matrix-vector products on GPU
    const index_t m = ls_cov_->K_rows_, n = ls_cov_->K_cols_;
    GEMV<scalar_t>::run(
        bls_handle_, CUBLAS_OP_N,
        m, n, 
        &ALPHA, ls_cov_->d_K_val_, m,
        d_vecB_, 1, &BETA, ls_cov_->d_rhs_bnd_, 1);
    GEMV<scalar_t>::run(
        bls_handle_,
        cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
        m, n,
        &ALPHA, ls_cov_->d_K_val_, m,
        ls_cov_->d_rhs_bnd_, 1, &BETA, d_temp_, 1);

    scalar_t b_dot_tmp = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
        d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
        &b_dot_tmp,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    scalar_t alpha = absNew/b_dot_tmp;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecX_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    alpha *= -1;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));

    RESD2 = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        &RESD2,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    const real_t re_resd2 = get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2);
    if (verbose) std::cout << "re_resd2 after iter " << i << ": " << re_resd2 << std::endl;
    if (residual) residual->emplace_back(re_resd2*TOL*TOL/threshold);
    if (re_resd2 < threshold) {
      if (verbose) std::cout << "early ending at iter " << i << "!!" << std::endl;
      break;
    }

    // solve A*z = resd
    if (preconditioned) {
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &ALPHA, A_, resd_, &BETA, temp_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                  cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                  &ALPHA, A_, temp_, &BETA, z_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    } else {
      CHECK_CUDA(cudaMemcpy(d_z_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
    }

    scalar_t absOld = absNew;
    absNew = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_z_, cuda_data_trait<scalar_t>::dataType, 1,
        &absNew,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);
    
    scalar_t beta = absNew/absOld;
    CHECK_CUBLAS(cublasScalEx(bls_handle_, n_, &beta,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &ALPHA,
                              cuda_data_trait<scalar_t>::dataType,
                              d_z_,    cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
  }
      
  CHECK_CUDA(cudaMemcpy(h_x, d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  if(verbose){
    std::cout << "pcg done!!!!!!!!!!" << std::endl;
  }
  return std::make_pair(i, sqrt(
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2)/
      get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2)));
}

template <enum PDE_TYPE pde>
gpu_simpl_klchol<pde>::pcg_ret_t
gpu_simpl_klchol<pde>::pcg_VpL_guesss(const scalar_t *h_rhs_bnd,
                               const index_t size_bnd,
                               scalar_t *h_x,
                               const bool preconditioned,
                               const index_t maxits,
                               const real_t TOL,
                               const bool verbose,
                               std::vector<real_t> *residual)
{
  const int d = pde_trait<pde>::d;
  ASSERT(ls_cov_->d_K_val_ != NULL);
  ASSERT(size_bnd == ls_cov_->K_rows_);

  const index_t threads_num = 256;
  const index_t blocks_num = (npts_+threads_num-1)/threads_num;
  const index_t sec_blocks_num = (npts_*num_sec_+threads_num-1)/threads_num;
  const scalar_t ALPHA = 1, BETA = 0;

  if (residual) {
    residual->clear();
    residual->reserve(2*maxits);
    residual->emplace_back(1.0);
  }

  if (ls_cov_->d_rhs_bnd_ == NULL) {
    CHECK_CUDA(cudaMalloc((void **)&ls_cov_->d_rhs_bnd_, size_bnd*sizeof(scalar_t)));
  }

  CHECK_CUDA(cudaMemcpy(d_resd_, h_rhs_bnd, ls_cov_->K_cols_*sizeof(scalar_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_vecX_, 0, n_*sizeof(scalar_t)));

  scalar_t RHS2 = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      &RHS2,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);
  if (get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2) < 1e-16) {
    CHECK_CUDA(cudaMemset(h_x, 0, n_*sizeof(scalar_t)));
    return std::make_pair(index_t(0), real_t(0));
  }

  const real_t threshold = TOL*TOL*get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RHS2);
  if(verbose){
    std::cout << "pcg debug!!!!!!!!!!\nthreshold: " << threshold << std::endl;
  }

  // solve A*b = resd
  if (preconditioned) {
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ALPHA, A_, resd_, &BETA, temp_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                &ALPHA, A_, temp_, &BETA, b_,
                                cuda_data_trait<scalar_t>::dataType,
                                CUSPARSE_SPMV_ALG_DEFAULT, d_work_));    
  } else {
    CHECK_CUDA(cudaMemcpy(d_vecB_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
  }

  scalar_t absNew = 0;
  DOTP<scalar_t>::run(
      bls_handle_, n_,
      d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
      d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
      &absNew,
      cuda_data_trait<scalar_t>::dataType,
      cuda_data_trait<scalar_t>::dataType);

  index_t i = 0;
  scalar_t RESD2 = 0;

  while (i < maxits) {
    ++i;
    
    // Compute matrix-vector products on GPU
    const index_t m = ls_cov_->K_rows_, n = ls_cov_->K_cols_;
    GEMV<scalar_t>::run(
        bls_handle_, CUBLAS_OP_N,
        m, n, 
        &ALPHA, ls_cov_->d_K_val_, m,
        d_vecB_, 1, &BETA, ls_cov_->d_rhs_bnd_, 1);
    GEMV<scalar_t>::run(
        bls_handle_,
        cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::blas_opType,
        m, n,
        &ALPHA, ls_cov_->d_K_val_, m,
        ls_cov_->d_rhs_bnd_, 1, &BETA, d_temp_, 1);

    scalar_t b_dot_tmp = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
        d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
        &b_dot_tmp,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    scalar_t alpha = absNew/b_dot_tmp;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecX_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    alpha *= -1;
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &alpha,
                              cuda_data_trait<scalar_t>::dataType,
                              d_temp_, cuda_data_trait<scalar_t>::dataType, 1,
                              d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));

    RESD2 = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        &RESD2,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);

    const real_t re_resd2 = get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2);
    if (verbose) std::cout << "re_resd2 after iter " << i << ": " << re_resd2 << std::endl;
    if (residual) residual->emplace_back(re_resd2*TOL*TOL/threshold);
    if (re_resd2 < threshold) {
      if (verbose) std::cout << "early ending at iter " << i << "!!" << std::endl;
      break;
    }

    // solve A*z = resd
    if (preconditioned) {
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &ALPHA, A_, resd_, &BETA, temp_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
      CHECK_CUSPARSE(cusparseSpMV(cus_handle_,
                                  cuda_adjoint_trait<std::is_floating_point<scalar_t>::value>::opType,
                                  &ALPHA, A_, temp_, &BETA, z_,
                                  cuda_data_trait<scalar_t>::dataType,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_work_));
    } else {
      CHECK_CUDA(cudaMemcpy(d_z_, d_resd_, n_*sizeof(scalar_t), cudaMemcpyDeviceToDevice));
    }

    scalar_t absOld = absNew;
    absNew = 0;
    DOTP<scalar_t>::run(
        bls_handle_, n_,
        d_resd_, cuda_data_trait<scalar_t>::dataType, 1,
        d_z_, cuda_data_trait<scalar_t>::dataType, 1,
        &absNew,
        cuda_data_trait<scalar_t>::dataType,
        cuda_data_trait<scalar_t>::dataType);
    
    scalar_t beta = absNew/absOld;
    CHECK_CUBLAS(cublasScalEx(bls_handle_, n_, &beta,
                              cuda_data_trait<scalar_t>::dataType,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
    CHECK_CUBLAS(cublasAxpyEx(bls_handle_, n_, &ALPHA,
                              cuda_data_trait<scalar_t>::dataType,
                              d_z_,    cuda_data_trait<scalar_t>::dataType, 1,
                              d_vecB_, cuda_data_trait<scalar_t>::dataType, 1,
                              cuda_data_trait<scalar_t>::dataType));
  }
      
  CHECK_CUDA(cudaMemcpy(h_x, d_vecX_, n_*sizeof(scalar_t), cudaMemcpyDeviceToHost));

  return std::make_pair(i, std::min(1.0f, (float)(std::sqrt(get_real<std::is_floating_point<scalar_t>::value, scalar_t>::run(RESD2)/threshold)/TOL)));
}
}