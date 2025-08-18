import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from core.controller import *
from core.KTAB_Mfs_spclf1d import *
from utils.meshcat_polys_meshio import *
from utils.save_utils import *
from models.utils.makeModels import makeModel_meta
import time
import os

# user settings
kirchhoff_offset = 0.0012
q_init = np.array([0.0, 0, -np.pi/10, 0.02, -0.03, -np.pi/4])
v_init = np.array([0.0, 0, 0, 0.0, 0, 0])
problem_name = "coinsPlanar_v5"
kirchhoff_offset_list = [kirchhoff_offset, kirchhoff_offset]
controller = Controller_grasp(policy=0)
dt = 0.01
T = 4.0
save_every = 2
rho_mfs = 1.2 # parameters for preconditioner
pcg_max_iter = 75 # max iterations for pcg
tol_r = 2.7e-3 # tolerance for pcg
drag_kind = 'none'
use_added_mass = True
indep = False
save_res = True
display_delay_factor = 6
rho_fluid = 998.0

model, collision_model, visual_model, list_for_compstJoints, _ = makeModel_meta(problem_name)

if visual_model is not None:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    viz.display(q_init)

# simulation setup
time_no_grad = []
time_grad = []
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
output_dir = './output/mfs/' + problem_name + '/' + time.strftime('%m-%d_%H-%M') + '/'
if save_res and not os.path.exists(output_dir):
    os.makedirs(output_dir)
if save_res:
    log_file_name = output_dir + 'log.txt'

# start simulation
for k in range(N):
    tic = time.time()
    if use_added_mass:
        time_start = time.time()
        Mf = kt.compute_kirchhoff_tensor_meta(q, v, renew_grad=True, indep=indep)
        time_end = time.time()
        print('time of compute_kirchhoff_tensor_meta: ', time_end - time_start)
        time_no_grad.append(kt.time_no_grad)
        time_grad.append(kt.time_grad_L)
        dMfdq = kt.grad_Mf_q_KDetached.detach().cpu().numpy()
        Mftderiv = np.einsum('ijk,k->ij', dMfdq, v) + kt.QR_deltaKRQ.detach().cpu().numpy()/dt

        bb = Mftderiv @ v
        bf = kt.grad_L_q.detach().cpu().numpy()
        drag = kt.drag_min.detach().cpu().numpy() * 0.0

    b = pin.rnea(model, data, q, v, a_0)
    M = pin.crba(model, data, q)
    L = 0.5 * np.dot(v, np.dot(M + Mf, v))

    tau_control = controller.control_sig(q, v)

    if use_added_mass:
        a = np.linalg.solve(M + Mf, bf - bb - b + tau_control + drag)
    else:
        a = np.linalg.solve(M, tau_control - b)

    v += a * dt
    q = pin.integrate(model, q, v * dt)  # Configuration integration
    pin.forwardKinematics(model, data, q)

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
    print('ellapsed in total: ',ellapsed)
    dt_sleep = max(0, dt*display_delay_factor - (ellapsed))
    time.sleep(dt_sleep)
    t += dt
