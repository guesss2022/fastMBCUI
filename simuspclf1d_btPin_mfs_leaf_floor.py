import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from core.controller import *
from core.KTAB_Mfs_spclf1d import *
from utils.meshcat_polys_meshio import *
from utils.save_utils import *
from utils.bt_utils import *
from utils.pin_utils import load_model_std
from models.utils.makeModels import makeModel_meta
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

# user settings
kirchhoff_offset = 0.0012
q_init = np.array([0.875, 0.7, -0.875, 1.57079633, -1.48])
v_init = np.array([-0., 0, 0, 0, 0])
problem_name = "leaf_v1x10"
kirchhoff_offset_list = [kirchhoff_offset*30, kirchhoff_offset*10, 0.21]
controller = Controller_leaf_v0(policy=2, scale=10)
dt = 0.01
T = 2.5
save_every = 1
rho_mfs = 1.15 # parameters for preconditioner
pcg_max_iter = 75 # max iterations for pcg
tol_r = 2.7e-3 # tolerance for pcg
mu = 0.65 # bullet friction coefficient
drag_kind = 'ILTdrag'
vis = 12.
use_added_mass = True
indep = False
save_res = True
display_delay_factor = 6
rho_fluid = 998.0

model, collision_model, visual_model, list_for_compstJoints = makeModel_meta(problem_name)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=False)
viz.loadViewerModel()
viz.display(q_init)

# bullet setup
physicsClientId = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0., 0., 0.)
p.setTimeStep(dt)
pinocchio_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model_path = os.path.join(pinocchio_model_dir, "example-robot-data/robots")
urdf_model_path = os.path.join(os.path.join(model_path, "guesss_description/robots"), problem_name + "_bt.urdf")
flags = p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
model_id = p.loadURDF(urdf_model_path, useFixedBase=True, flags=flags)
set_q_v(model_id, q_init, v_init)
p.setJointMotorControlArray(model_id, range(model.nv), p.VELOCITY_CONTROL, forces=np.zeros(model.nv).tolist())
for i in range(model.nv):
    p.changeDynamics(model_id, i, lateralFriction=mu)

# simulation setup
data = model.createData()
N = math.floor(T / dt)
t = 0.0
q = q_init
v = v_init
a_0 = np.zeros(model.nv)
model.gravity.linear *= (1-1/1.0)
viz.display(q)
if use_added_mass:
    kt = Kirchhoff_tensor_multi_Mfs(model, visual_model, kirchhoff_offset_list, rho_fluid,
                            problem_name, rho_mfs, pcg_max_iter, tol_r, drag_kind, list_for_compstJoints)

first_name = problem_name.split('_')[0]
v = MakeInitVelo(problem_name, q, v, use_added_mass, kt, indep, model, data, 1.0)

meshes = load_meshes(viz)
input('Press Enter to continue...')
time.sleep(1)
print('start simulation')
# saving directory
output_dir = './output/btPin_mfs_' + drag_kind + '/' + problem_name + '/' + time.strftime('%m-%d_%H-%M') + '/'
if save_res and not os.path.exists(output_dir):
    os.makedirs(output_dir)
if save_res:
    log_file_name = output_dir + 'log.txt'

for k in range(N):
    tic = time.time()
    if use_added_mass:
        time_start = time.time()
        Mf = kt.compute_kirchhoff_tensor_meta(q, v, renew_grad=True, indep=indep)
        time_end = time.time()
        print('time of compute_kirchhoff_tensor_meta: ', time_end - time_start)

        dMfdq = kt.grad_Mf_q_KDetached.detach().cpu().numpy().squeeze()
        if len(dMfdq.shape) == 2:
            dMfdq = dMfdq[:, :, None]
        Mftderiv = np.einsum('ijk,k->ij', dMfdq, v) + kt.QR_deltaKRQ.detach().cpu().numpy()/dt

        bb = Mftderiv @ v
        bf = kt.grad_L_q.detach().cpu().numpy()
        drag = kt.drag_min.detach().cpu().numpy() * vis
        drag[0] = 0

    b = pin.rnea(model, data, q, v, a_0)
    M = pin.crba(model, data, q)
    L = 0.5 * np.dot(v, np.dot(M + Mf, v))

    tau_control = controller.control_sig(q, v, M+Mf if use_added_mass else M)

    if use_added_mass: # force floor to be static!
        nv = model.nv
        M_eff = M + Mf
        M0 = M_eff[:nv-1, :nv-1]
        b_eff = bf - bb - b + tau_control + drag
        b0 = b_eff[:nv-1]
        a0 = np.linalg.solve(M0, b0)
        a = np.zeros_like(v)
        a[:nv-1] = a0
    else:
        nv = model.nv
        M0 = M[:nv-1, :nv-1]
        b_eff = -b + tau_control + drag
        b0 = b_eff[:nv-1]
        a0 = np.linalg.solve(M0, b0)
        a = np.zeros_like(v)
        a[:nv-1] = a0

    tau_bt = M @ a + b
    p.setJointMotorControlArray(model_id, range(model.nv), p.TORQUE_CONTROL, forces=tau_bt.tolist())
    p.stepSimulation()
    q, v = get_q_v(model_id)
    pin.forwardKinematics(model, data, q)
    toc = time.time()
    ellapsed = toc - tic
    print('ellapsed in total: ',ellapsed)
    viz.display(q)

    if save_res and k % save_every == 0:
        filename = output_dir + '{:05d}.obj'.format(k)
        update_and_save_to_obj(meshes, filename, viz)
        with open(log_file_name, 'a') as f:
            f.write('frame: ' + str(k) + '; q: ' + str(q) + '\n')
            f.write('frame: ' + str(k) + '; v: ' + str(v) + '\n')
            f.write('frame: ' + str(k) + '; a: ' + str(a) + '\n\n')
        with open(output_dir+'L_log.txt', 'a') as f:
            f.write('frame: ' + str(k) + '; L: ' + str(L) + '\n')
        with open(output_dir+'b_log.txt', 'a') as f:
            f.write('frame: ' + str(k) + '; -bb: ' + str(-bb) + '\n')
            f.write('frame: ' + str(k) + '; bf: ' + str(bf) + '\n')
            f.write('frame: ' + str(k) + '; b_fluid: ' + str(bf - bb) + '\n\n')
        with open(output_dir+'Meff_log.txt', 'a') as f:
            f.write('frame: ' + str(k) + '; Meff: \n' + str(M + Mf) + '\n')
    else:
        print('frame: ' + str(k))
    toc = time.time()
    ellapsed = toc - tic
    dt_sleep = max(0, dt*display_delay_factor - (ellapsed))
    time.sleep(dt_sleep)
    t += dt

p.disconnect()