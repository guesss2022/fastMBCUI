from utils.torch_utils import *
import torch
import pinocchio as pin
import numpy as np

def _make_R_from_rotvec(rotvec:np.ndarray, q:float, device, dtype):
    I = torch.eye(3, device=device, dtype=dtype)
    rrT = torch.tensor(np.outer(rotvec, rotvec), device=device, dtype=dtype)
    rx = torch.tensor([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]], [-rotvec[1], rotvec[0], 0]], dtype=dtype, device=device)
    return torch.cos(q) * I + torch.sin(q) * rx + (1 - torch.cos(q)) * rrT

def parse_joint_name(joint_name:str):
    if joint_name == 'JointModelRX':
        kind_id_ = 0
    elif joint_name == 'JointModelRY':
        kind_id_ = 1
    elif joint_name == 'JointModelRZ':
        kind_id_ = 2
    elif joint_name == 'JointModelPX':
        kind_id_ = 3
    elif joint_name == 'JointModelPY':
        kind_id_ = 4
    elif joint_name == 'JointModelPZ':
        kind_id_ = 5
    elif joint_name == 'JointModelCX':
        kind_id_ = 6
    elif joint_name == 'JointModelCY':
        kind_id_ = 7
    elif joint_name == 'JointModelCZ':
        kind_id_ = 8
    elif joint_name == 'JointModelPlanarYZ':
        kind_id_ = 9
    elif joint_name == 'JointModelPlanarYZV2':
        kind_id_ = 10
    elif joint_name == 'JointModelFreeFlyer':
        kind_id_ = -1
    else:
        raise Exception("Joint Type not Implemented")
    return kind_id_

def handle_joint(q:torch.Tensor, gqind:int, r:torch.Tensor, R_r:torch.Tensor, kind_id:int, device, dtype):
    if kind_id == 0:
        nq = 1
        nv = 1
        r_now = r
        rot_vec_ = np.array([1.0, 0, 0])
        trans_vec_ = np.array([0.0, 0, 0])
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 1:
        nq = 1
        nv = 1
        r_now = r
        rot_vec_ = np.array([0.0, 1, 0])
        trans_vec_ = np.array([0.0, 0, 0])
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 2:
        nq = 1
        nv = 1
        r_now = r
        rot_vec_ = np.array([0.0, 0, 1])
        trans_vec_ = np.array([0.0, 0, 0])
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 3:
        nq = 1
        nv = 1
        rot_vec_ = np.array([0.0, 0, 0])
        trans_vec_ = np.array([1.0, 0, 0])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        R_r_now = R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 4:
        nq = 1
        nv = 1
        rot_vec_ = np.array([0.0, 0, 0])
        trans_vec_ = np.array([0.0, 1, 0])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        R_r_now = R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 5:
        nq = 1
        nv = 1
        rot_vec_ = np.array([0.0, 0, 0])
        trans_vec_ = np.array([0.0, 0, 1])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        R_r_now = R_r
        Qp_now = torch.tensor(np.block([trans_vec_, rot_vec_]), dtype=dtype, device=device)[:,None]
    elif kind_id == 6:
        nq = 2
        nv = 2
        rot_vec_ = np.array([1.0, 0, 0])
        trans_vec_ = np.array([1.0, 0, 0])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind+1], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([[trans_vec_[:,None],np.zeros((3,1))],[np.zeros((3,1)),rot_vec_[:,None]]]), dtype=dtype, device=device)
    elif kind_id == 7:
        nq = 2
        nv = 2
        rot_vec_ = np.array([0.0, 1, 0])
        trans_vec_ = np.array([0.0, 1, 0])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind+1], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([[trans_vec_[:,None],np.zeros((3,1))],[np.zeros((3,1)),rot_vec_[:,None]]]), dtype=dtype, device=device)
    elif kind_id == 8:
        nq = 2
        nv = 2
        rot_vec_ = np.array([0.0, 0, 1])
        trans_vec_ = np.array([0.0, 0, 1])
        r_now = r + R_r @ torch.tensor(trans_vec_, dtype=dtype, device=device) * q[gqind]
        rot_vec = R_r.detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q[gqind+1], device, dtype)
        R_r_now = R_q @ R_r
        Qp_now = torch.tensor(np.block([[trans_vec_[:,None],np.zeros((3,1))],[np.zeros((3,1)),rot_vec_[:,None]]]), dtype=dtype, device=device)
    elif kind_id == 9:
        nq = 3
        nv = 3
        rot_vec_ = np.array([1.0, 0, 0])
        trans_vec1_ = np.array([0.0, 1, 0])
        trans_vec2_ = np.array([0.0, 0, 1])
        R_q = _make_R_from_rotvec(rot_vec_, q[gqind], device, dtype)
        R_r_now = R_r @ R_q
        r_now = r + R_r_now @ (torch.tensor(trans_vec1_, dtype=dtype, device=device) * q[gqind+1] + \
                                 torch.tensor(trans_vec2_, dtype=dtype, device=device) * q[gqind+2])
        Qp_now = torch.tensor(np.block([[np.zeros((3,1)),trans_vec1_[:,None],trans_vec2_[:,None]],[rot_vec_[:,None],np.zeros((3,1)),np.zeros((3,1))]]), dtype=dtype, device=device)
        Qp_now[1,0] = -q[gqind+2]
        Qp_now[2,0] = q[gqind+1]
    elif kind_id == 10:
        nq = 3
        nv = 3
        rot_vec_ = np.array([1.0, 0, 0])
        trans_vec1_ = np.array([0.0, 1, 0])
        trans_vec2_ = np.array([0.0, 0, 1])
        R_q = _make_R_from_rotvec(rot_vec_, q[gqind+2], device, dtype)
        R_r_now = R_r @ R_q
        r_now = r + R_r @ (torch.tensor(trans_vec1_, dtype=dtype, device=device) * q[gqind] + \
                                 torch.tensor(trans_vec2_, dtype=dtype, device=device) * q[gqind+1])
        Qp_now_ = torch.tensor(np.block([[trans_vec1_[:,None],trans_vec2_[:,None],np.zeros((3,1))],[np.zeros((3,1)),np.zeros((3,1)),rot_vec_[:,None]]]), dtype=dtype, device=device)
        Qp_now = torch.zeros(6, 3, dtype=dtype, device=device)
        Qp_now[:3,:] = R_q.T @ Qp_now_[:3,:]
        Qp_now[3:,:] = Qp_now_[3:,:]
    elif kind_id == -1:
        nq = 7
        nv = 6
        r_now = r + R_r @ q[gqind:gqind+3]
        R_r_now = R_r @ compute_quat2matrix(q[gqind+3:gqind+7])
        # R_r_now = compute_quat2matrix(q[gqind+3:gqind+7]) @ R_r
        Qp_now = torch.tensor(np.eye(6), dtype=dtype, device=device)
    else:
        raise Exception("Joint Type not Implemented")

    return r_now, R_r_now, nq, nv, Qp_now


### legacy
def compute_oMi_Qp_torch(model: pin.Model, q: np.ndarray, device='cpu', dtype=torch.float32,
                      grad_joint_id=None, R_grad_buffer=None, y_grad_buffer=None, Qp_grad_buffer=None):
    r = []
    R_r = []
    kind_id = []
    parents = []
    for joint_id in range(1, model.njoints):
        joint = model.joints[joint_id]
        if joint.shortname() == 'JointModelRX':
            kind_id_ = 0
        elif joint.shortname() == 'JointModelRY':
            kind_id_ = 1
        elif joint.shortname() == 'JointModelRZ':
            kind_id_ = 2
        elif joint.shortname() == 'JointModelFreeFlyer':
            kind_id_ = -1
        else:
            raise Exception("Unknown joint type")
        if joint_id == 1:
            assert kind_id_ == -1
        else:
            assert kind_id_ == 0 or kind_id_ == 1 or kind_id_ == 2
        kind_id.append(kind_id_)
        r.append(torch.tensor(model.jointPlacements[joint_id].translation, dtype=dtype, device=device))
        R_r.append(torch.tensor(model.jointPlacements[joint_id].rotation, dtype=dtype, device=device))
        parents.append(model.parents[joint_id])
    R = []
    y = []
    Qp = []
    R.append(torch.eye(3, device=device))
    y.append(torch.zeros(3, device=device))
    Qp.append(torch.tensor(np.block([np.eye(6),np.zeros((6, len(q)-7))]), dtype=dtype, device=device))
    q_torch = torch.tensor(q[7:], device=device, requires_grad=True)
    for i in range(model.njoints-2):
        parent = parents[i+1] - 1
        y.append(y[parent] + R[parent] @ r[i+1])
        if kind_id[i+1] == 0:
            rot_vec_ = np.array([1.0, 0, 0])
        elif kind_id[i+1] == 1:
            rot_vec_ = np.array([0.0, 1, 0])
        else:
            rot_vec_ = np.array([0.0, 0, 1])
        rot_vec = R_r[i+1].detach().cpu().numpy() @ rot_vec_
        R_q = _make_R_from_rotvec(rot_vec, q_torch[i], device, dtype)
        R.append(R[parent] @ R_q @ R_r[i+1])

        Qp_ = torch.zeros(6, len(q)-1, device=device, dtype=dtype)
        Qp_pre = Qp[parent]
        Qp_[0:3, :] = Qp_pre[0:3, :] + torch.cross(Qp_pre[3:6, :], y[i+1][:,None] - y[parent][:,None], dim=0)
        Qp_[3:6, :] = Qp_pre[3:6, :]
        Qp_[3:6, i+6] += R[i+1] @ torch.tensor(rot_vec_, dtype=dtype, device=device)
        Qp.append(Qp_)

    if grad_joint_id is not None:
        R_ = R[grad_joint_id]
        y_ = y[grad_joint_id]
        Qp_ = Qp[grad_joint_id]
        for i in range(3):
            for j in range(3):
                R_[i, j].backward(retain_graph=True)
                R_grad_buffer[i, j] = q_torch.grad.clone()
                q_torch.grad.zero_()
            y_[i].backward(retain_graph=True)
            y_grad_buffer[i] = q_torch.grad.clone()
            q_torch.grad.zero_()
        for i in range(6):
            for j in range(len(q)-1):
                Qp_[i, j].backward(retain_graph=True)
                Qp_grad_buffer[i, j] = q_torch.grad.clone()
                q_torch.grad.zero_()

    return R, y, Qp
### END legacy