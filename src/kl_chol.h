// Copyright (C) 2021 Jiong Chen
// This code is licensed under GPL v3.

#ifndef KL_CHOL_H
#define KL_CHOL_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <Eigen/Sparse>
#include <type_traits>
#include <vector>

#include "macro.h"
#include "kernels.h"

namespace klchol {

// ----------------------- kernel matrices for least squares ---------------------
template <enum PDE_TYPE pde>
struct cov_assembler
{
  typedef typename pde_trait<pde>::scalar_t scalar_t;
  typedef typename pde_trait<pde>::index_t  index_t;
  typedef typename pde_trait<pde>::real_t   real_t;

  // number of observation points and boundary points
  // can be different for equivalent source method
  cov_assembler()
  {
    CHECK_CUBLAS(cublasCreate(&handle_));
  }
  
  virtual ~cov_assembler()
  {
    // source, boundary points
    if ( d_src_xyz_ ) CHECK_CUDA(cudaFree(d_src_xyz_));
    if ( d_bnd_xyz_ ) CHECK_CUDA(cudaFree(d_bnd_xyz_));

    // source, normals
    if ( d_src_nxyz_ ) CHECK_CUDA(cudaFree(d_src_nxyz_));
    if ( d_bnd_nxyz_ ) CHECK_CUDA(cudaFree(d_bnd_nxyz_));

    if ( d_rhs_src_ ) CHECK_CUDA(cudaFree(d_rhs_src_));
    if ( d_rhs_bnd_ ) CHECK_CUDA(cudaFree(d_rhs_bnd_));

    if ( d_K_val_ )   CHECK_CUDA(cudaFree(d_K_val_));

    if ( d_aux_ )     CHECK_CUDA(cudaFree(d_aux_));

    if ( handle_ ) { CHECK_CUBLAS(cublasDestroy(handle_)); }
  }

  void set_source_pts(const index_t n_pts, const real_t *h_xyz, const real_t *h_nxyz=NULL)
  {
    std::cout << "n_src_pts=" << n_pts << std::endl;
    n_src_pts_ = n_pts;

    // points
    if ( d_src_xyz_ == NULL ) {
      CHECK_CUDA(cudaMalloc((void **)&d_src_xyz_, 3*n_src_pts_*sizeof(real_t)));
    }
    CHECK_CUDA(cudaMemcpy(d_src_xyz_, h_xyz, 3*n_src_pts_*sizeof(real_t), cudaMemcpyHostToDevice));

    // normals
    if ( d_src_nxyz_ == NULL ) {
      CHECK_CUDA(cudaMalloc((void **)&d_src_nxyz_, 3*n_src_pts_*sizeof(real_t)));
    }
    CHECK_CUDA(cudaMemset(d_src_nxyz_, 0, 3*n_src_pts_*sizeof(real_t)));
    if ( h_nxyz != NULL ) {
      CHECK_CUDA(cudaMemcpy(d_src_nxyz_, h_nxyz, 3*n_src_pts_*sizeof(real_t), cudaMemcpyHostToDevice));
    }
  }
  
  void set_bound_pts(const index_t n_pts, const real_t *h_xyz, const real_t *h_nxyz=NULL)
  {
    std::cout << "n_bnd_pts=" << n_pts << std::endl;
    n_bnd_pts_ = n_pts;

    // boundary points
    if ( d_bnd_xyz_ == NULL ) {
      CHECK_CUDA(cudaMalloc((void **)&d_bnd_xyz_, 3*n_bnd_pts_*sizeof(real_t)));
    }   
    CHECK_CUDA(cudaMemcpy(d_bnd_xyz_, h_xyz, 3*n_bnd_pts_*sizeof(real_t), cudaMemcpyHostToDevice));

    // boundary normals
    if ( d_bnd_nxyz_ == NULL ) {
      CHECK_CUDA(cudaMalloc((void **)&d_bnd_nxyz_, 3*n_bnd_pts_*sizeof(real_t)));
    }
    CHECK_CUDA(cudaMemset(d_bnd_nxyz_, 0, 3*n_bnd_pts_*sizeof(real_t)));
    if ( h_nxyz != NULL ) {
      CHECK_CUDA(cudaMemcpy(d_bnd_nxyz_, h_nxyz, 3*n_bnd_pts_*sizeof(real_t), cudaMemcpyHostToDevice));
    }
  }

  // store G (n_bnd x n_src) instead of G^T*G, long rectangular matrix
  void build_cov_mat_LS(const KERNEL_TYPE ker_type,
                        const real_t *h_ker_aux,
                        const index_t length);
  void build_cov_mat_LS(const double* M_p_data, const int K_rows, const int K_cols, 
    const KERNEL_TYPE ktype, const double* param, const int param_size);
  void build_cov_mat_LS_float(const float* M_p_data, const int K_rows, const int K_cols, 
    const KERNEL_TYPE ktype, const float* param, const int param_size);
  index_t rows() const
  {
    return K_rows_;
  }
  index_t cols() const
  {
    return K_cols_;    
  }
  void K(scalar_t *Kval) const;

  // G^T*b
  void build_cov_rhs_LS(const scalar_t *rhs_bnd, const index_t size_bnd,
                        scalar_t *rhs_src, const index_t size_src);

  cublasHandle_t handle_ = NULL;

  // source points, number, coords, and normals
  index_t n_src_pts_ = 0;
  real_t *d_src_xyz_  = NULL;
  real_t *d_src_nxyz_ = NULL;

  // boundary points for equivalent sources
  index_t n_bnd_pts_ = 0;
  real_t *d_bnd_xyz_  = NULL;
  real_t *d_bnd_nxyz_ = NULL;

  index_t K_rows_, K_cols_ = 0;
  scalar_t *d_K_val_   = NULL;
  scalar_t *d_rhs_bnd_ = NULL;
  scalar_t *d_rhs_src_ = NULL;

  const index_t AUX_SIZE_ = 16;
  real_t *d_aux_ = NULL;
};

// ----------------------- cuda solver -------------------------

template <enum PDE_TYPE pde>
class gpu_simpl_klchol
{
 public:
  typedef typename pde_trait<pde>::scalar_t scalar_t;
  typedef typename pde_trait<pde>::index_t  index_t;
  typedef typename pde_trait<pde>::real_t   real_t;
  typedef std::pair<index_t, real_t>        pcg_ret_t;
  typedef cov_assembler<pde>                covariance_t;
  
  // sparsity for pattern
  // [npts] number of points
  // [dim]  dimension
  // [nnz]  nnz of BIE matrix
  gpu_simpl_klchol(const index_t npts, const index_t ker_dim, const index_t num_sec);
  virtual ~gpu_simpl_klchol();

  // intialization
  void set_source_points(const index_t npts, const real_t *h_xyz, const real_t *h_nxyz=NULL);
  void set_kernel(const KERNEL_TYPE ker_type,
                  const real_t *h_ker_aux,
                  const index_t length);

  virtual void set_supernodes(const index_t n_super,
                              const index_t n_pts,
                              const index_t *h_super_ptr,
                              const index_t *h_super_ind,
                              const index_t *h_super_parent)
  {
  }
  virtual void set_sppatt(const index_t n, const index_t nnz,
                          const index_t *h_ptr,
                          const index_t *h_ind);

  real_t theta_nnz() const { return TH_nnz_; }

  // compute
  virtual void assemble();
  virtual void factorize();
  virtual void compute();
  virtual void get_factor(Eigen::SparseMatrix<scalar_t> &L) const;
  virtual real_t memory() const;

  void solve(const scalar_t *h_rhs, scalar_t *h_x);

  pcg_ret_t pcg_guesss(const scalar_t *h_rhs_bnd,
                      const index_t size_bnd,
                      scalar_t *h_x,
                      const bool preconditioned,
                      const index_t maxits=1000,
                      const real_t TOL=1e-6,
                      const bool verbose=false,
                      std::vector<real_t> *residual=nullptr);

  pcg_ret_t pcg_VpL_guesss(const scalar_t *h_rhs_bnd,
                      const index_t size_bnd,
                      scalar_t *h_x,
                      const bool preconditioned,
                      const index_t maxits=1000,
                      const real_t TOL=1e-6,
                      const bool verbose=false,
                      std::vector<real_t> *residual=nullptr);
  
  void evalKx(scalar_t *Kx) const;

  // prediction from training set
  void set_target_points(const index_t npts, const real_t *h_xyz);  
  void predict(const int impulse_dim,
               const scalar_t *src, const index_t src_num,
               scalar_t       *res, const index_t res_num) const;

  void train_and_predict(const scalar_t *h_rhs,
                         const index_t n_train,
                         scalar_t *h_y,
                         const index_t threads_num=128);
  // --- predict directly using Cholesky factorization
  void chol_predict(const scalar_t *h_rhs,
                    const index_t n_train,
                    scalar_t *h_y);

  template <class Vec> 
  void kernel(const Vec &x, const Vec &y, const real_t *param, scalar_t *K) const {
    gf_summary<pde, scalar_t, real_t>::run(static_cast<int>(ker_type_), &x[0], &y[0], param, K);
  }
  void kernel(const index_t i, const index_t j, scalar_t *K) const {
    if ( ls_cov_->d_K_val_ == NULL ) {
      return;
    }
    ASSERT(0);
  }
  
  scalar_t* get_TH_val() { return d_TH_val_; }
  void set_TH_val(scalar_t* val) { d_TH_val_ = val; }
  
  scalar_t* get_val() { return d_val_; }
  void set_val(scalar_t* val) { d_val_ = val; }
  
  index_t* get_TH_ptr() { return d_TH_ptr_; }
  void set_TH_ptr(index_t* ptr) { d_TH_ptr_ = ptr; }
  
  real_t get_TH_nnz() const { return TH_nnz_; }
  void set_TH_nnz(real_t nnz) { TH_nnz_ = nnz; }
  
  index_t get_nnz() const { return nnz_; }
  void set_nnz(index_t nnz) { nnz_ = nnz; }

  covariance_t *ls_cov_ = NULL;

 protected:
  cusparseHandle_t cus_handle_ = NULL;
  cublasHandle_t bls_handle_ = NULL;
  
  // number of points and dimensionality
  index_t npts_, ker_dim_, n_, num_sec_;

  // matrix A from BIE and Cholesky factor of inv(A)
  index_t nnz_ = 0;
  index_t  *d_ptr_ = NULL;
  index_t  *d_ind_ = NULL;
  scalar_t *d_val_ = NULL;
  cusparseSpMatDescr_t A_ = NULL;

  // sectioned results
  scalar_t *d_sec_eval_ = NULL;

  // vector b and x
  scalar_t *d_vecB_ = NULL, *d_vecX_ = NULL;
  scalar_t *d_resd_ = NULL, *d_temp_ = NULL, *d_z_ = NULL;
  cusparseDnVecDescr_t b_ = NULL, x_ = NULL;
  cusparseDnVecDescr_t resd_ = NULL, temp_ = NULL, z_ = NULL;
  cusparseSpSVDescr_t spsv_desc_ = NULL;
  
  // kernel type
  const int AUX_LENGTH_ = 16;
  KERNEL_TYPE ker_type_;
  real_t *d_ker_aux_ = NULL;
  
  // spatial coordinates, might be reordered  
  real_t *d_xyz_ = NULL;
  real_t *d_nxyz_ = NULL;

  // for prediction
  index_t npts_pred_ = 0;
  real_t   *d_pred_xyz_ = NULL;
  scalar_t *d_pred_f_ = NULL;
  
  // neighbors and local LHS
  real_t TH_nnz_ = 0;
  index_t  *d_TH_ptr_ = NULL;
  scalar_t *d_TH_val_ = NULL;

  // working buffer
  void *d_work_ = NULL;
  size_t buff_sz_;
};

template <enum PDE_TYPE pde>
class gpu_super_klchol : public gpu_simpl_klchol<pde>
{
 public:
  typedef typename pde_trait<pde>::scalar_t scalar_t;
  typedef typename pde_trait<pde>::index_t  index_t;
  typedef typename pde_trait<pde>::real_t   real_t;
  
  gpu_super_klchol(const index_t npts, const index_t ker_dim, const index_t sec_num);
  ~gpu_super_klchol();

  // initialization
  void set_supernodes(const index_t n_super,
                      const index_t n_pts,
                      const index_t *h_super_ptr,
                      const index_t *h_super_ind,
                      const index_t *h_super_parent);
  void set_sppatt(const index_t n, const index_t nnz,
                  const index_t *h_ptr,
                  const index_t *h_ind);  

  // supernodal factorization
  void assemble();
  void factorize();
  void compute();
  real_t memory() const;

 private:
  // gpu data
  index_t n_super_ = 0;
  index_t *d_super_ptr_ = NULL;
  index_t *d_super_ind_ = NULL;
  index_t *d_super_parent_ = NULL;

  // cpu data
  index_t *h_super_ptr_ = NULL;
  index_t *h_super_ind_ = NULL;
};

}
#endif
