#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace mfs_torch {

TORCH_LIBRARY(mfs_torch, m) {
    m.def("VpL_fast_float(Tensor M, Tensor pL, Tensor Vs_s, float rho, int pcg_max_iter, float tol_r, bool timing_verbose=False, bool solve_verbose=False) -> Tensor");
    m.def("MinvF_float_fastGrad(Tensor M, Tensor F, Tensor Vs_s, float rho, int pcg_max_iter, float tol_r, bool timing_verbose=False, bool solve_verbose=False) -> Tensor");
    }
}