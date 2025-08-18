// Copyright (C) 2025  Tianhong Gao

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "src/preprocess.h"
#include "src/kl_chol.h"
#include "src/gputimer.h"

using namespace std;
using namespace Eigen;
using namespace klchol;

namespace mfs_torch {

// Add cache variables for float version
static at::Tensor s_d_TH_val_cache_float;
static at::Tensor s_d_val_cache_float;

at::Tensor MinvF_float(const at::Tensor& M, const at::Tensor& F, const at::Tensor& Vs_s,
        const double rho, const int64_t pcg_max_iter, const double tol_r, const bool timing_verbose,
        const bool solve_verbose) {
    TORCH_CHECK(F.dim() == 2, "F must be a 2D tensor");
    TORCH_CHECK(M.size(0) == F.size(0), "M and F must have the same first dimension");
    TORCH_CHECK(M.size(1) == Vs_s.size(0), "M and Vs_s must have compatible dimensions");
    TORCH_CHECK(Vs_s.size(1) == 3, "Vs_s must be 3d points");
    TORCH_CHECK(M.dtype() == at::kFloat);
    TORCH_CHECK(F.dtype() == at::kFloat);
    TORCH_CHECK(Vs_s.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(M.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(F.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(Vs_s.device().type() == at::DeviceType::CPU);

    GpuTimer timer_tot, timer_solver;
    timer_tot.start();

    timer_solver.start();
    at::Tensor F_contig = F.contiguous();
    at::Tensor Vs_s_contig = Vs_s.contiguous();
    
    int64_t rows = Vs_s_contig.size(0);
    int64_t cols = 3;
    Map<Eigen::Matrix<float, -1, -1, 1>> Vsrcs(Vs_s_contig.data_ptr<float>(), rows, cols);
    Eigen::Matrix<float, -1, -1, 1> Vsrcs_float = Vsrcs;
    Eigen::Matrix<double, -1, -1, 1> Vsrcs_double = Vsrcs.cast<double>();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(prep): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    fps_sampler fps(Vsrcs_double);
    fps.compute('F');
    fps.reorder_geometry(Vsrcs_double);
    fps.debug();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(fps_sampler): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Obtain permutation matrix P
    timer_solver.start();
    const auto& P = fps.P();
    
    // Create index tensor for P.T operation
    at::Tensor indices = torch::empty({rows}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    // For P.T, we need the inverse permutation
    std::vector<int64_t> inverse_indices(rows);
    for (int i = 0; i < rows; i++) {
        int64_t j = P.indices()(i);
        inverse_indices[j] = i;
    }
    for (int i = 0; i < rows; i++) {
        indices_ptr[i] = inverse_indices[i];
    }
    indices = indices.to(at::kCUDA);
    at::Tensor M_p = torch::index_select(M, 1, indices);
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(permutation): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Create result tensor - use float type
    int64_t n_cols_F = F.size(1);
    at::Tensor result_float = torch::empty({rows, n_cols_F}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    result_float.fill_(0);
    result_float = result_float.contiguous();
    
    klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_NM_3D_SL;
    float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
    int   NUM_SUPERNODES = 0;
    int   MAX_SUPERNODE_SIZE = 2048;
    int   NUM_SEC = 16;
    const size_t N = Vsrcs.rows();
    const std::vector<size_t> GROUP{0, N};

    std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON_FLOAT>> super_solver;
    super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON_FLOAT>(N, 1, NUM_SEC));

    Eigen::VectorXf PARAM = Eigen::VectorXf::Zero(8); 
    PARAM[2] = (float)rho; 
    PARAM[7] = (float)KTYPE;
    Eigen::SparseMatrix<double> PATT, SUP_PATT;
    VectorXi sup_ptr, sup_ind, sup_parent;
    super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
    
    timer_solver.start();
    int K_rows = M_p.size(0);
    int K_cols = M_p.size(1);
    at::Tensor M_p_t = M_p.t().contiguous();
    const float* M_p_data = M_p_t.data_ptr<float>();
    
    super_solver->ls_cov_->build_cov_mat_LS_float(M_p_data, K_rows, K_cols, KTYPE, PARAM.data(), PARAM.size());
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(build_cov_mat): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    fps.simpl_sparsity((float)rho, 1, PATT);
    fps.aggregate(1, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
    fps.super_sparsity(1, PATT, sup_parent, SUP_PATT);
    NUM_SUPERNODES = sup_ptr.size()-1;
    SIMPL_NNZ = PATT.nonZeros();
    SUPER_NNZ = SUP_PATT.nonZeros();
    PARAM[6] = 1.0*SUPER_NNZ/SUP_PATT.size();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(pattern): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
    super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
    THETA_NNZ = super_solver->theta_nnz();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(setup): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    super_solver->compute();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(compute): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Save computed results to float cache
    timer_solver.start();
    
    // Get sizes of d_TH_val_ and d_val_
    size_t TH_nnz_size = super_solver->get_TH_nnz();
    size_t nnz_size = super_solver->get_nnz();
    
    // Get pointers
    float* d_TH_val_ptr = (float*)super_solver->get_TH_val();
    float* d_val_ptr = (float*)super_solver->get_val();
    
    // Create cache tensors
    s_d_TH_val_cache_float = torch::empty(TH_nnz_size, torch::dtype(torch::kFloat).device(torch::kCUDA));
    s_d_val_cache_float = torch::empty(nnz_size, torch::dtype(torch::kFloat).device(torch::kCUDA));
    
    // Copy data to caches
    CHECK_CUDA(cudaMemcpy(s_d_TH_val_cache_float.data_ptr<float>(), d_TH_val_ptr, 
                          TH_nnz_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(s_d_val_cache_float.data_ptr<float>(), d_val_ptr, 
                         nnz_size * sizeof(float), cudaMemcpyDeviceToDevice));
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(cache): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Perform PCG solve for each column
    timer_solver.start();
    // Transpose F and result matrices so columns become rows for contiguous access
    at::Tensor F_t = F_contig.t().contiguous();
    at::Tensor result_t = torch::empty({n_cols_F, rows}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    
    // Each row now corresponds to the original column for direct access
    for (int64_t i = 0; i < n_cols_F; i++) {
        const float* F_row_ptr = F_t.data_ptr<float>() + i * F_t.size(1);
        float* result_row_ptr = result_t.data_ptr<float>() + i * result_t.size(1);
        super_solver->pcg_guesss(F_row_ptr, (size_t)F_t.size(1),
            result_row_ptr, true, pcg_max_iter, tol_r, solve_verbose);
    }
    
    // Transpose back to original shape
    at::Tensor result_ = result_t.to(torch::kFloat).t().contiguous();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(pcg): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    // Create index tensor for P.T operation
    at::Tensor indices_ = torch::empty({rows}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* indices_ptr_ = indices_.data_ptr<int64_t>();
    for (int i = 0; i < rows; i++) {
        int64_t j = P.indices()(i);
        indices_ptr_[i] = j;
    }
    indices_ = indices_.to(at::kCUDA);
    at::Tensor result = torch::index_select(result_, 0, indices_);
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(perm_back): " << timer_solver.elapsed() << " ms" << std::endl;

    timer_tot.stop();
    if (timing_verbose) {
        std::cout << "TIME(total): " << timer_tot.elapsed() << " ms" << std::endl;
    }

    return result;
}

at::Tensor VpL_fast_float(const at::Tensor& M, const at::Tensor& pL, const at::Tensor& Vs_s,
        const double rho, const int64_t pcg_max_iter, const double tol_r, const bool timing_verbose,
        const bool solve_verbose) {
    TORCH_CHECK(pL.dim() == 2, "pL must be a 2D tensor");
    TORCH_CHECK(M.size(1) == pL.size(0), "M and pL must have compatible dimensions");
    TORCH_CHECK(M.size(1) == Vs_s.size(0), "M and Vs_s must have compatible dimensions");
    TORCH_CHECK(Vs_s.size(1) == 3, "Vs_s must be 3d points");
    TORCH_CHECK(M.dtype() == at::kFloat);
    TORCH_CHECK(pL.dtype() == at::kFloat);
    TORCH_CHECK(Vs_s.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(M.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(pL.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(Vs_s.device().type() == at::DeviceType::CPU);
    
    // Check that float caches exist
    TORCH_CHECK(s_d_TH_val_cache_float.defined() && s_d_val_cache_float.defined(), 
                "Float cache tensors are not defined. Please run MinvF_float first.");

    GpuTimer timer_tot, timer_solver;
    timer_tot.start();

    timer_solver.start();
    at::Tensor pL_contig = pL.contiguous();
    at::Tensor Vs_s_contig = Vs_s.contiguous();
    
    int64_t rows = Vs_s_contig.size(0);
    int64_t cols = 3;
    Map<Eigen::Matrix<float, -1, -1, 1>> Vsrcs(Vs_s_contig.data_ptr<float>(), rows, cols);
    Eigen::Matrix<float, -1, -1, 1> Vsrcs_float = Vsrcs;
    Eigen::Matrix<double, -1, -1, 1> Vsrcs_double = Vsrcs.cast<double>();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(prep): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    fps_sampler fps(Vsrcs_double);
    fps.compute('F');
    fps.reorder_geometry(Vsrcs_double);
    fps.debug();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(fps_sampler): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Obtain permutation matrix P
    timer_solver.start();
    const auto& P = fps.P();
    
    // Create index tensor for P.T operation
    at::Tensor indices = torch::empty({rows}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* indices_ptr = indices.data_ptr<int64_t>();
    // For P.T, we need the inverse permutation
    std::vector<int64_t> inverse_indices(rows);
    for (int i = 0; i < rows; i++) {
        int64_t j = P.indices()(i);
        inverse_indices[j] = i;
    }
    for (int i = 0; i < rows; i++) {
        indices_ptr[i] = inverse_indices[i];
    }
    indices = indices.to(at::kCUDA);
    at::Tensor M_p = torch::index_select(M, 1, indices);
    
    // Reorder pL accordingly
    at::Tensor pL_p = torch::index_select(pL_contig, 0, indices);
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(permutation): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Create result tensor
    int64_t n_cols_pL = pL.size(1);
    at::Tensor result_float = torch::empty({rows, n_cols_pL}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    result_float.fill_(0);
    result_float = result_float.contiguous();
    
    klchol::KERNEL_TYPE KTYPE = klchol::KERNEL_TYPE::LAPLACE_NM_3D_SL;
    float SIMPL_NNZ = 0, SUPER_NNZ = 0, THETA_NNZ = 0;
    int   NUM_SUPERNODES = 0;
    int   MAX_SUPERNODE_SIZE = 2048;
    int   NUM_SEC = 16;
    const size_t N = Vsrcs.rows();
    const std::vector<size_t> GROUP{0, N};

    std::unique_ptr<klchol::gpu_simpl_klchol<PDE_TYPE::POISSON_FLOAT>> super_solver;
    super_solver.reset(new klchol::gpu_super_klchol<PDE_TYPE::POISSON_FLOAT>(N, 1, NUM_SEC));

    Eigen::VectorXf PARAM = Eigen::VectorXf::Zero(8); 
    PARAM[2] = (float)rho; 
    PARAM[7] = (float)KTYPE;
    Eigen::SparseMatrix<double> PATT, SUP_PATT;
    VectorXi sup_ptr, sup_ind, sup_parent;
    super_solver->set_kernel(KTYPE, PARAM.data(), PARAM.size());
    
    timer_solver.start();
    int K_rows = M_p.size(0);
    int K_cols = M_p.size(1);
    at::Tensor M_p_t = M_p.t().contiguous();
    const float* M_p_data = M_p_t.data_ptr<float>();
    
    super_solver->ls_cov_->build_cov_mat_LS_float(M_p_data, K_rows, K_cols, KTYPE, PARAM.data(), PARAM.size());
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(build_cov_mat): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    fps.simpl_sparsity((float)rho, 1, PATT);
    fps.aggregate(1, GROUP, PATT, 1.5, sup_ptr, sup_ind, sup_parent, MAX_SUPERNODE_SIZE);
    fps.super_sparsity(1, PATT, sup_parent, SUP_PATT);
    NUM_SUPERNODES = sup_ptr.size()-1;
    SIMPL_NNZ = PATT.nonZeros();
    SUPER_NNZ = SUP_PATT.nonZeros();
    PARAM[6] = 1.0*SUPER_NNZ/SUP_PATT.size();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(pattern): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    super_solver->set_supernodes(sup_ptr.size()-1, sup_ind.size(), sup_ptr.data(), sup_ind.data(), sup_parent.data());
    super_solver->set_sppatt(SUP_PATT.rows(), SUP_PATT.nonZeros(), SUP_PATT.outerIndexPtr(), SUP_PATT.innerIndexPtr());
    THETA_NNZ = super_solver->theta_nnz();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(setup): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Use cached float results and skip compute() step
    timer_solver.start();
    
    // Get pointers
    float* d_TH_val_ptr = (float*)super_solver->get_TH_val();
    float* d_val_ptr = (float*)super_solver->get_val();
    
    // Copy cached data to solver
    CHECK_CUDA(cudaMemcpy(d_TH_val_ptr, s_d_TH_val_cache_float.data_ptr<float>(), 
                          s_d_TH_val_cache_float.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_val_ptr, s_d_val_cache_float.data_ptr<float>(), 
                         s_d_val_cache_float.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(copy_cache): " << timer_solver.elapsed() << " ms" << std::endl;
    
    // Perform PCG solve for each column
    timer_solver.start();
    // Transpose pL and result matrices so columns become rows for contiguous access
    at::Tensor pL_t = pL_p.t().contiguous();
    at::Tensor result_t = torch::empty({n_cols_pL, rows}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    
    // Each row now corresponds to the original column for direct access
    for (int64_t i = 0; i < n_cols_pL; i++) {
        const float* pL_row_ptr = pL_t.data_ptr<float>() + i * pL_t.size(1);
        float* result_row_ptr = result_t.data_ptr<float>() + i * result_t.size(1);
        super_solver->pcg_VpL_guesss(pL_row_ptr, (size_t)K_rows,
            result_row_ptr, true, pcg_max_iter, tol_r, solve_verbose);
    }
    
    // Transpose back to original shape
    at::Tensor result_ = result_t.t().contiguous();
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(pcg): " << timer_solver.elapsed() << " ms" << std::endl;
    
    timer_solver.start();
    // Create index tensor for P.T operation
    at::Tensor indices_ = torch::empty({rows}, torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* indices_ptr_ = indices_.data_ptr<int64_t>();
    for (int i = 0; i < rows; i++) {
        int64_t j = P.indices()(i);
        indices_ptr_[i] = j;
    }
    indices_ = indices_.to(at::kCUDA);
    at::Tensor result = torch::index_select(result_, 0, indices_);
    timer_solver.stop();
    if (timing_verbose) std::cout << "TIME(perm_back): " << timer_solver.elapsed() << " ms" << std::endl;

    timer_tot.stop();
    if (timing_verbose) {
        std::cout << "TIME(total): " << timer_tot.elapsed() << " ms" << std::endl;
    }

    return result;
}

TORCH_LIBRARY_IMPL(mfs_torch, CUDA, m) {
    m.impl("VpL_fast_float", &VpL_fast_float);
    m.impl("MinvF_float_fastGrad", &MinvF_float);
}

}