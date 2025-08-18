#ifndef KL_KERNELS_H
#define KL_KERNELS_H

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <memory>

#include "traits.h"

namespace klchol {

#define CUDA_PI 3.1415926535897932
#define SQRT_3  1.7320508075688772
#define R_4PI   0.07957747154594767

// -------------------- kernel functions ----------------------------
enum KERNEL_TYPE
{
  LAPLACE_NM_3D_SL=0
};

static const char *KERNEL_LIST = "lap_nm_3d_sl\0\0";

template <typename scalar_t, typename real_t>
__device__ __host__ __forceinline__
void laplace_nm_3d_sl(const real_t *x,
                   const real_t *y,
                   const real_t *n_x,
                   const real_t *n_y,
                   const real_t *p,
                   scalar_t     *G)
{
  const real_t r[3] = {*x-*y, *(x+1)-*(y+1), *(x+2)-*(y+2)};

  const real_t reg_r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2],
      reg_r1 = sqrt(reg_r2), reg_r3 = reg_r1*reg_r2;
  real_t rnx = r[0]*n_x[0] + r[1]*n_x[1] + r[2]*n_x[2];

  *G = 1.0/reg_r3*rnx;
}

// ------------------- register kernels to PDE ------------------------------
template <enum PDE_TYPE pde, typename scalar_t, typename real_t>
struct gf_summary;

// Float versions of gf_summary specializations
template <typename scalar_t, typename real_t>
struct gf_summary<PDE_TYPE::POISSON_FLOAT, scalar_t, real_t>
{
  __device__ __host__ __forceinline__
  static void run(const int      id,
                  const real_t   *x,
                  const real_t   *y,
                  const real_t   *n_x,
                  const real_t   *n_y,
                  const real_t   *p,
                  scalar_t       *G)
  {
    switch ( id ) {
      case KERNEL_TYPE::LAPLACE_NM_3D_SL:
        laplace_nm_3d_sl(x, y, n_x, n_y, p, G);
        break;
      default:
        printf("# unsupported kernel!\n");        
    }
  }

  // FOR LEAST-SQUARE
  __device__ __forceinline__
  static void run(const int i, const int j,
                  const int n_bnd,
                  const scalar_t *K_mat,
                  scalar_t *G)
  {
    // G_{ij} = K(:,i)^T K(:,j)
    *G = thrust::inner_product(
        thrust::device,
        K_mat+i*n_bnd,
        K_mat+(i+1)*n_bnd,
        K_mat+j*n_bnd,
        scalar_t(0),
        thrust::plus<scalar_t>(),
        thrust::multiplies<scalar_t>());
  }
};
}
#endif
