# Copyright (C) 2025  Tianhong Gao

from utils.torch_utils import *
from utils.kinematics_torch import *
import core.mfs_torch_py as mfs
import meshio
import torch
import time
import pinocchio as pin

class Kirchhoff_tensor_multi_Mfs:
    def __init__(self, model: pin.Model, visual_model, offset_list, rho_fluid, problem_name:str,
                 rho_mfs, pcg_max_iter, tol_r, drag_kind:str="none", list_for_compstJoints=None, testing=False): 
        
        num_links = model.njoints - 1
        if visual_model is None:
            testing = True
        else:
            assert len(offset_list) == num_links and len(visual_model.geometryObjects) == num_links, \
                'offset_list and visual_model.geometryObjects should have the same length: num_links'
        self.num_links = num_links
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('running on device:', self.device)
        self.dtype = torch.float32
        self.rho_fluid = rho_fluid
        self.drag_kind = drag_kind
        self.time_no_grad = 0.0
        self.time_grad_L = 0.0
        self.rho_mfs = rho_mfs
        self.pcg_max_iter = pcg_max_iter
        self.tol_r = tol_r

        if not testing:
            gs = visual_model.geometryObjects
            filenames = [g.meshPath for g in gs]
            y0_s = [g.placement.translation for g in gs]
            R0_s = [g.placement.rotation for g in gs]
            meshs = [meshio.read(filename) for filename in filenames]
            pointss = [mesh.points.copy() for mesh in meshs]
            V0_s = [(R0 @ points.T).T + y0 for R0, y0, points in zip(R0_s, y0_s, pointss)]
            F_s = [mesh.cells[0].data for mesh in meshs]
            
            self.V0_s = [torch.tensor(V0, dtype=self.dtype).to(self.device) for V0 in V0_s]
            self.F_s = [torch.tensor(F, dtype=torch.int32).to(self.device) for F in F_s]

            self.num_faces_s = [F.shape[0] for F in F_s]
            self.num_faces_total = sum(self.num_faces_s)
            self.num_source_total = sum([V0.shape[0] for V0 in self.V0_s])

            N0_s = [compute_normal_vertex(V0, F) for V0, F in zip(V0_s, F_s)]
            self.S0_s = [V0 - offset * torch.tensor(N0, dtype=self.dtype).to(self.device) \
                        for V0, N0, offset in zip(self.V0_s, N0_s, offset_list)]
            self.N0_s = [torch.tensor(N0, dtype=self.dtype).to(self.device) for N0 in N0_s]

        self.njoints = model.njoints
        self.nv = model.nv
        self.nq = model.nq
        self.grad_Mf_q_KDetached = torch.zeros(self.nv, self.nv, self.nq, dtype=self.dtype, device=self.device)
        self.grad_Mf_q_DEBUG = torch.zeros(self.nv, self.nv, self.nq, dtype=self.dtype, device=self.device)
        self.grad_L_q = torch.zeros(self.nq, dtype=self.dtype, device=self.device)
        self.K_last = None
        self.QR_deltaKRQ = torch.zeros(self.nv, self.nv, dtype=self.dtype, device=self.device)
        self.deltaQRKRQ = torch.zeros(self.nv, self.nv, dtype=self.dtype, device=self.device)
        self.drag_min = None
        
        self.r = []
        self.R_r = []
        self.kind_id = []
        self.parents = []
        for joint_id in range(1, model.njoints):
            joint = model.joints[joint_id]
            if joint.shortname() == 'JointModelComposite':
                assert list_for_compstJoints is not None and len(list_for_compstJoints[joint_id]) > 0, \
                    'list_for_compstJoints requirement not satisfied'
                kind_id_ = parse_joint_name(list_for_compstJoints[joint_id])
            else:
                kind_id_ = parse_joint_name(joint.shortname())
            self.kind_id.append(kind_id_)
            self.r.append(torch.tensor(model.jointPlacements[joint_id].translation, dtype=self.dtype, device=self.device))
            self.R_r.append(torch.tensor(model.jointPlacements[joint_id].rotation, dtype=self.dtype, device=self.device))
            self.parents.append(model.parents[joint_id])


    def compute_kirchhoff_tensor_meta(self, q:np.ndarray, v:np.ndarray, renew_grad = True, indep = False):
        assert indep == False, 'indep not supported for now'
        return self.compute_kirchhoff_tensor(q, v, renew_grad)

    def compute_kirchhoff_tensor(self, q:np.ndarray, v:np.ndarray, renew_grad = True):
        time_start = time.time()

        q_torch = torch.tensor(q, dtype=self.dtype, device=self.device, requires_grad=True)
        v_torch = torch.tensor(v, dtype=self.dtype, device=self.device)
        R_s, y_s, Qps = self.compute_oMi_Qp_torch(q_torch, device=self.device)
        Qp = torch.cat(tuple(Qps), dim=0)


        V_s = [(R @ V0.T).T + y for R, V0, y in zip(R_s, self.V0_s, y_s)]
        S_s = [(R @ S0.T).T + y for R, S0, y in zip(R_s, self.S0_s, y_s)]
        S = torch.cat(tuple(S_s), dim=0)
        C_s = [compute_face_centers(V, F) for V, F in zip(V_s, self.F_s)]
        Cr_s = [C - y for C, y in zip(C_s, y_s)]
        # FL_s = [compute_FL_matrix_onC_noMulArea(V, F, Cr) for V, F, Cr in zip(V_s, self.F_s, Cr_s)]
        FL_s = [compute_FL_matrix(V, F, Cr) for V, F, Cr in zip(V_s, self.F_s, Cr_s)]
        FL = torch.cat(tuple([torch.cat(tuple([(FL if i == j else torch.zeros((num_faces, 6), dtype=self.dtype, device=self.device)) \
                                   for j in range(self.num_links)]), dim=1) \
                  for FL, num_faces, i in zip(FL_s, self.num_faces_s, range(self.num_links))]),dim=0)
        

        IQ6_np = np.block([[np.zeros((3,3)), np.eye(3)],[np.eye(3), np.zeros((3,3))]])
        IQ_np = np.block([[(IQ6_np if j == i else np.zeros((6,6))) for j in range(self.num_links)] for i in range(self.num_links)])
        IQ = torch.tensor(IQ_np, dtype=self.dtype, device=self.device)
        
        F = FL @ IQ @ Qp
        M = torch.cat(tuple([compute_solid_angle(V, F, S) for V, F in zip(V_s, self.F_s)]),dim=0)
        if renew_grad:
            # strengths = M.pinverse() @ FL
            strengths = mfs.ops.MinvF_float_fastGrad(M, F, S.detach().cpu(), self.rho_mfs, self.pcg_max_iter, self.tol_r, timing_verbose=False, solve_verbose=False)
            del S_s, FL_s, FL, M
            torch.cuda.empty_cache()
        else:
            result_ = torch.linalg.lstsq(M, FL)
            strengths = result_.solution
        SL_s = [compute_single_layer(S, C) for C in C_s]
        phi_s = [SL @ strengths for SL in SL_s]
        Q_s = [compute_quadrature(F, V, Cr) for F, V, Cr in zip(self.F_s, V_s, Cr_s)]
        Ks_ = torch.cat(tuple([Q @ phi * self.rho_fluid for Q, phi in zip(Q_s, phi_s)]), dim=0)

        Mf = Qp.T @ IQ @ Ks_
        L = 0.5 * v_torch @ Mf @ v_torch

        print('Mf_cal_done using multi_dependent!!!')
        time_end = time.time()
        print('Time elapsed computing noGrad:', time_end - time_start)
        self.time_no_grad = time_end - time_start
        
        if renew_grad:
            time_start = time.time()
            L.backward(retain_graph=True)
            self.grad_L_q = q_torch.grad.clone()
            q_torch.grad.zero_()
            time_end = time.time()
            print('Time elapsed computing L-Grad:', time_end - time_start)
            self.time_grad_L = time_end - time_start

        R6ups = [torch.cat((R, torch.zeros((3, 3), dtype=self.dtype, device=self.device)), dim=1) for R in R_s]
        R6downs = [torch.cat((torch.zeros((3, 3), dtype=self.dtype, device=self.device), R), dim=1) for R in R_s]
        R6s = [torch.cat((R6up, R6down), dim=0) for R6up, R6down in zip(R6ups, R6downs)]
        R6n = torch.cat(tuple([torch.cat(tuple([R6i if i == j else torch.zeros((6, 6), dtype=self.dtype, device=self.device) \
                    for j in range(self.num_links)]),dim=1) for i, R6i in zip(range(self.num_links), R6s)]), dim=0)

        K_ = R6n.T @ IQ @ Ks_
        K_ = K_.detach()
        if self.K_last is not None:
            self.QR_deltaKRQ = Qp.T @ R6n @ (K_-self.K_last)

        self.K_last = K_.clone()
        self.Mf_last = Mf.clone()

        Mf_KDetached = Qp.T @ R6n @ K_

        if renew_grad:
            time_start = time.time()
            for i in range(self.nv):
                for j in range(self.nv):
                    Mf_KDetached[i, j].backward(retain_graph=True)
                    self.grad_Mf_q_KDetached[i, j] = q_torch.grad.clone()
                    q_torch.grad.zero_()
            time_end = time.time()
            print('Time elapsed computing Mf-Grad:', time_end - time_start)

        # start computing drag
        if self.drag_kind == 'none':
            self.drag_min = torch.zeros(self.nv, dtype=self.dtype, device=self.device)
        else:
            if self.drag_kind == 'ILdrag':
                drag_s = [compute_aggregate_ILdrag(V, F, Cr) for V, F, Cr in zip(V_s, self.F_s, Cr_s)]
            elif self.drag_kind == 'ILTdrag':
                drag_s = [compute_aggregate_ILTdrag(V, F, Cr) for V, F, Cr in zip(V_s, self.F_s, Cr_s)]
            elif self.drag_kind == 'ILNdrag':
                drag_s = [compute_aggregate_ILNdrag(V, F, Cr) for V, F, Cr in zip(V_s, self.F_s, Cr_s)]
            drag_max = torch.cat(tuple([torch.cat(tuple([drag if i == j else torch.zeros((6, 6), dtype=self.dtype, device=self.device) \
                        for j in range(self.num_links)]),dim=1) for i, drag in zip(range(self.num_links), drag_s)]), dim=0)
            self.drag_min = Qp.T @ IQ @ drag_max @ IQ @ Qp @ v_torch

        del V_s, R_s, y_s, Qps, Qp, S, C_s, Cr_s, SL_s
        del strengths, phi_s, Q_s, Ks_, IQ, q_torch, v_torch
        del L, R6ups, R6downs, R6s, R6n, K_, Mf_KDetached
        torch.cuda.empty_cache()

        return Mf.detach().cpu().numpy()

    def compute_oMi_Qp_torch(self, q: torch.Tensor, device='cpu', dtype=torch.float32):
        R = []
        y = []
        Qp = []
        q_torch = q
        gqind = 0
        gvind = 0
        for i in range(self.njoints-1):
            parent = self.parents[i] - 1

            r_now, R_r_now, nq, nv, Qp_now = handle_joint(q_torch, gqind, self.r[i], self.R_r[i], self.kind_id[i], device, dtype)

            if parent == -1:
                y_parent = torch.zeros(3, device=device, dtype=dtype)
                R_parent = torch.eye(3, device=device, dtype=dtype)
                Qp_parent = torch.zeros(6, self.nv, device=device, dtype=dtype)
            else:
                y_parent = y[parent]
                R_parent = R[parent]
                Qp_parent = Qp[parent]
            
            y.append(y_parent + R_parent @ r_now)
            R.append(R_parent @ R_r_now)

            Qp_ = torch.zeros(6, self.nv, device=device, dtype=dtype)
            Qp_[0:3, :] = Qp_parent[0:3, :] + torch.cross(Qp_parent[3:6, :], y[i][:,None] - y_parent[:,None], dim=0)
            Qp_[3:6, :] = Qp_parent[3:6, :]
            Qp_[0:3, gvind:gvind+nv] += R[i] @ Qp_now[0:3, :]
            Qp_[3:6, gvind:gvind+nv] += R[i] @ Qp_now[3:6, :]
            Qp.append(Qp_)

            gqind += nq
            gvind += nv

        return R, y, Qp