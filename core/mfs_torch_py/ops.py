import torch
from torch import Tensor

__all__ = ["MinvF_float_fastGrad"]

### MinvF_float_fastGrad
def MinvF_float_fastGrad(M: Tensor, F: Tensor, Vs_s: Tensor, rho: float, pcg_max_iter: int, tol_r: float,
        timing_verbose: bool=False, solve_verbose: bool=False) -> Tensor:
    """Performs M.pinv @ F"""
    return torch.ops.mfs_torch.MinvF_float_fastGrad.default(M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose)

@torch.library.register_fake("mfs_torch::MinvF_float_fastGrad")
def _(M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose):
    torch._check(M.shape[0] == F.shape[0])
    torch._check(M.shape[1] == Vs_s.shape[0])
    torch._check(Vs_s.shape[1] == 3)
    torch._check(M.dtype == torch.float)
    torch._check(F.dtype == torch.float)
    torch._check(Vs_s.dtype == torch.float)
    torch._check(M.device == F.device)
    return torch.empty([Vs_s.shape[0], F.shape[1]], dtype=torch.float, device=M.device)

def _setup_context_MinvF_float_fastGrad(ctx, inputs, output):
    M, F, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose = inputs
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_M = M
        saved_F = F
        saved_Vs_s = Vs_s
        saved_A = output
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
    ctx.save_for_backward(saved_M, saved_F, saved_Vs_s, saved_A)

def _backward_MinvF_float_fastGrad(ctx, grad):
    grad_M, grad_F = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        M, F, Vs_s, A = ctx.saved_tensors
        rho = ctx.rho
        pcg_max_iter = ctx.pcg_max_iter
        tol_r = ctx.tol_r
        timing_verbose = ctx.timing_verbose
        solve_verbose = ctx.solve_verbose
        VpL = torch.ops.mfs_torch.VpL_fast_float.default(M, grad, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose)
    if ctx.needs_input_grad[0]:
        LV = VpL.T
        W = A @ LV
        grad_M = -M @ (W + W.T) + F @ LV
    if ctx.needs_input_grad[1]:
        grad_F = M @ VpL
    return grad_M, grad_F, None, None, None, None, None, None

torch.library.register_autograd("mfs_torch::MinvF_float_fastGrad",
    _backward_MinvF_float_fastGrad, setup_context=_setup_context_MinvF_float_fastGrad)

@torch.library.register_fake("mfs_torch::VpL_fast_float")
def _(M, pL, Vs_s, rho, pcg_max_iter, tol_r, timing_verbose, solve_verbose):
    torch._check(M.shape[1] == pL.shape[0])
    torch._check(M.shape[1] == Vs_s.shape[0])
    torch._check(Vs_s.shape[1] == 3)
    torch._check(M.dtype == torch.float)
    torch._check(pL.dtype == torch.float)
    torch._check(Vs_s.dtype == torch.float)
    torch._check(M.device == pL.device)
    return torch.empty([Vs_s.shape[0], pL.shape[1]], dtype=torch.float, device=M.device)