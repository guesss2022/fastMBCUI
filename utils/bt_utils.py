import numpy as np
import pybullet as p

def get_q_v(model_id):
    num_joints = p.getNumJoints(model_id)
    joint_states = p.getJointStates(model_id, range(num_joints))
    q_ = np.array([joint_state[0] for joint_state in joint_states])
    v_ = np.array([joint_state[1] for joint_state in joint_states])
    return q_, v_

def set_q_v(model_id, q_, v_):
    num_joints = p.getNumJoints(model_id)
    for i in range(num_joints):
        p.resetJointState(model_id, i, q_[i], targetVelocity=v_[i])