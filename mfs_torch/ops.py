# Copyright (C) 2025  Tianhong Gao

import torch
from torch import Tensor

__all__ = ["test_add", "Minvb", "MinvF"]

def test_add(a: Tensor, b: Tensor) -> Tensor:
    """Performs a + b"""
    return torch.ops.mfs_torch.test_add.default(a, b)

@torch.library.register_fake("mfs_torch::test_add")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

def Minvb(M: Tensor, b: Tensor, Vs_s: Tensor, rho: float, pcg_max_iter: int, tol_r: float,
          timing_verbose: bool=False, solve_verbose: bool=False) -> Tensor:
    """Performs M.pinv @ b"""
    return torch.ops.mfs_torch.Minvb.default(M, b, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose)

@torch.library.register_fake("mfs_torch::Minvb")
def _(M, b, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose):
    # Check that b is a 1D tensor
    torch._check(b.dim() == 1)
    torch._check(M.shape[0] == b.shape[0])
    torch._check(M.shape[1] == Vs_s.shape[0])
    torch._check(Vs_s.shape[1] == 3)
    torch._check(M.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(Vs_s.dtype == torch.float)
    torch._check(M.device == b.device)
    # Return shape is a 1D tensor of length Vs_s.shape[0]
    return torch.empty(Vs_s.shape[0], dtype=torch.float, device=M.device)

def MinvF(M: Tensor, F: Tensor, Vs_s: Tensor, rho: float, pcg_max_iter: int, tol_r: float,
          timing_verbose: bool=False, solve_verbose: bool=False) -> Tensor:
    """Performs M.pinv @ F"""
    return torch.ops.mfs_torch.MinvF.default(M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose)

@torch.library.register_fake("mfs_torch::MinvF")
def _(M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose):
    torch._check(M.shape[0] == F.shape[0])
    torch._check(M.shape[1] == Vs_s.shape[0])
    torch._check(Vs_s.shape[1] == 3)
    torch._check(M.dtype == torch.float)
    torch._check(F.dtype == torch.float)
    torch._check(Vs_s.dtype == torch.float)
    torch._check(M.device == F.device)
    # Return shape is a 2D tensor of shape [Vs_s.shape[0], F.shape[1]]
    return torch.empty([Vs_s.shape[0], F.shape[1]], dtype=torch.float, device=M.device)

def _setup_context_MinvF(ctx, inputs, output):
    M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose = inputs
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_M = M
        saved_F = F
        saved_Vs_s = Vs_s
        saved_A = output
        # Store non-tensor data as attributes on ctx
        ctx.rho = rho
        ctx.pcg_max_iter = pcg_max_iter
        ctx.tol_r = tol_r
        ctx.timing_verbose = timing_verbose
        ctx.solve_verbose = solve_verbose
    else:
        saved_M = None
        saved_F = None
        saved_Vs_s = None
        saved_A = None
    # Only save tensor-type data
    ctx.save_for_backward(saved_M, saved_F, saved_Vs_s, saved_A)

def _backward_MinvF(ctx, grad):
    print('debug!!!!!!!!!!!!!!\ngrad: ', grad.device, '\ndebug!!!!!!!!!!!!!!')
    grad_M, grad_F = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        M, F, Vs_s, A = ctx.saved_tensors
        rho = ctx.rho
        pcg_max_iter = ctx.pcg_max_iter
        tol_r = ctx.tol_r
        timing_verbose = ctx.timing_verbose
        solve_verbose = ctx.solve_verbose
        VpL = torch.ops.mfs_torch.VpL.default(M, grad, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose)
    if ctx.needs_input_grad[0]:
        LV = VpL.T
        W = A @ LV
        grad_M = -M @ (W + W.T) + F @ LV
    if ctx.needs_input_grad[1]:
        grad_F = M @ VpL
    return grad_M, grad_F, None, None, None, None, None, None

torch.library.register_autograd("mfs_torch::MinvF",
    _backward_MinvF, setup_context=_setup_context_MinvF)

@torch.library.register_fake("mfs_torch::VpL")
def _(M, pL, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose):
    torch._check(M.shape[1] == pL.shape[0])
    torch._check(M.shape[1] == Vs_s.shape[0])
    torch._check(Vs_s.shape[1] == 3)
    torch._check(M.dtype == torch.float)
    torch._check(pL.dtype == torch.float)
    torch._check(Vs_s.dtype == torch.float)
    torch._check(M.device == pL.device)
    # Return shape is a 2D tensor of shape [Vs_s.shape[0], pL.shape[1]]
    return torch.empty([Vs_s.shape[0], pL.shape[1]], dtype=torch.float, device=M.device)