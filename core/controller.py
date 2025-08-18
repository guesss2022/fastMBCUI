import numpy as np
import math
import pinocchio as pin
import torch

def MakeInitVelo(problem_name, q, v, use_added_mass, kt, indep, model, data, solid_mass_factor):
    problem_name_split = problem_name.split('_')
    first_name = problem_name_split[0]
    last_name = problem_name_split[-1]
    if use_added_mass and first_name == "coinsPlanar" and last_name in ["v1", "v2", "v4", "v5"]:
        print('coinsPlanar called MakeInitVelo automatically...\nmodify in controller if you want to...')
        Mf = kt.compute_kirchhoff_tensor_meta(q, v, renew_grad=True, indep=indep)
        _, __, Qps = kt.compute_oMi_Qp_torch(torch.tensor(q, device = 'cuda', dtype = torch.float32),device = 'cuda')
        Qp = torch.cat(tuple(Qps), dim=0).detach().cpu().numpy()
        mom_target = Qp.T @ np.array([0.,0,-6.47e-3*0.08,0,0,0,0,0,0,0,0,0])
        M = pin.crba(model, data, q) * solid_mass_factor
        v = np.linalg.solve(Mf + M, mom_target)
        print('v made: ', v)

    return v

def _pid(error, integral, pre_error, kp, ki, kd, dt):
    integral = integral + error * dt
    derivative = (error - pre_error) / dt
    signal = kp * error + ki * integral + kd * derivative
    return signal, integral, error    

class Controller_leaf_v0:
    def __init__(self, policy, scale=1.0):
        self.scale = scale
        self.control_state = 0
        self.policy = policy
        self.z_buffer = 0.0
        self.pid_integral = 0.0
        self.pid_pre_error = 0.0

    def control_sig(self, q, v, Meff):
        tau_control = np.zeros((len(v)))

        if self.policy == 0: # policy 0: let the ball accelerate down until a specified speed
            # print(v[0])
            if v[0] > -0.19 * self.scale and self.control_state==0:
                tau_control[0] = -0.19 * self.scale ** 4
            elif self.control_state==0:
                self.control_state=1

        if self.policy == 1: # policy 1: based on policy0, let the ball accelerate down until a specified height then stop it.
            dt = 0.01
            atz = -0.0418 * self.scale
            print('control_state: ', self.control_state)

            if self.control_state == 0:
                tau_control[0] = -0.19 * self.scale ** 4
                if v[0] <= -0.19 * self.scale:
                    self.control_state = 1
                if q[0] < atz:
                    self.control_state = 2
            elif self.control_state == 1:
                if q[0] < atz:
                    self.control_state = 2
            elif self.control_state == 2:
                Mm1 = np.linalg.inv(Meff)
                tau_control[0] = -v[0]/dt/Mm1[0,0]
                self.control_state = 3
                self.z_buffer = q[0]
            elif self.control_state == 3:
                tau_control[0], self.pid_integral, self.pid_pre_error = _pid(self.z_buffer - q[0], self.pid_integral, self.pid_pre_error, 0.005*self.scale**3, 0.005*self.scale**3, 0.007*self.scale**3, dt)

        if self.policy == 2: # policy 2: based on policy1, change to collision with floor.
            dt = 0.01
            atz = -0.0343 * self.scale
            print('control_state: ', self.control_state)

            if self.control_state == 0:
                tau_control[0] = -0.19 * self.scale ** 4 * (148./156.)**2
                if q[0] < atz:
                    self.control_state = 2
            elif self.control_state == 2:
                Mm1 = np.linalg.inv(Meff)
                tau_control[0] = -v[0]/dt/(Mm1[0,0]-Mm1[0,4]*Mm1[4,0]/Mm1[4,4])
                self.control_state = 3
                self.z_buffer = q[0]
            elif self.control_state == 3:
                tau_control[0], self.pid_integral, self.pid_pre_error = _pid(self.z_buffer - q[0], self.pid_integral, self.pid_pre_error, 0.005*self.scale**3, 0.005*self.scale**3, 0.007*self.scale**3, dt)
        
        return tau_control

class Controller_grasp:
    def __init__(self, policy, pos_firstFinger = 0, scale=1.0):
        self.control_state = 0
        self.policy = policy
        self.pos_firstFinger = pos_firstFinger
        self.scale = scale

    def control_sig(self, q, v):
        tau_control = np.zeros((len(v)))

        if self.policy == 0: # policy 0: default let them move freely
            pass

        elif self.policy == 1: # policy 1: grasp locally
            if self.control_state == 0:
                tau_control[self.pos_firstFinger] = 0.04 * self.scale**4
                if np.abs(v[self.pos_firstFinger]) > 0.02 * self.scale:
                    self.control_state = 1
            elif self.control_state == 1:
                tau_control[self.pos_firstFinger] = -0.0
            tau_control[self.pos_firstFinger+1] = -tau_control[self.pos_firstFinger]

        elif self.policy == 2: # policy 2: # chase and grasp
            if self.control_state == 0:
                tau_control[self.pos_firstFinger] = 0.04 * self.scale**4
                if np.abs(v[self.pos_firstFinger]) > 0.012 * self.scale:
                    self.control_state = 1
            elif self.control_state == 1:
                tau_control[self.pos_firstFinger] = 0.04 * self.scale**4
                tau_control[0] = 0.085 * self.scale**4
                if np.abs(v[self.pos_firstFinger]) > 0.018 * self.scale:
                    self.control_state = 2
            elif self.control_state == 2:
                tau_control[self.pos_firstFinger] = 0.0
                tau_control[0] = 0.0
            tau_control[self.pos_firstFinger+1] = -tau_control[self.pos_firstFinger]

        elif self.policy == 3: # policy 3: chase faster and grasp
            if self.control_state == 0:
                tau_control[self.pos_firstFinger] = 0.04 * self.scale**4
                if np.abs(v[self.pos_firstFinger]) > 0.008 * self.scale:
                    self.control_state = 1
            elif self.control_state == 1:
                tau_control[self.pos_firstFinger] = 0.04 * self.scale**4
                tau_control[0] = 0.1 * self.scale**4
                if np.abs(v[self.pos_firstFinger]) > 0.02 * self.scale:
                    self.control_state = 2
            elif self.control_state == 2:
                tau_control[self.pos_firstFinger] = 0.0
                tau_control[0] = 0.0
            tau_control[self.pos_firstFinger+1] = -tau_control[self.pos_firstFinger]

        return tau_control