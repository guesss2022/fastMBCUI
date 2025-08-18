import pinocchio as pin
from utils.pin_utils import *
from scipy.spatial.transform import Rotation as R
import numpy as np
def makeModel_meta(problem_name: str):
    problem_name_split = problem_name.split('_')
    first_name = problem_name_split[0]
    if first_name == 'coinsPlanar':
        return makeModel_coinsPlanar(problem_name)
    elif first_name == 'grasp':
        return makeModel_grasp(problem_name)
    elif first_name == 'leaf':
        return makeModel_leaf(problem_name)
    else:
        raise ValueError('problem not implemented')

def makeModel_coinsPlanar(problem_name):
    problem_name_split = problem_name.split('_')
    last_name = problem_name_split[-1]
    if last_name in ['v5']:
        urdf_filename = "coinsPlanar_v1"

    model_, collision_model, visual_model = load_model_std(urdf_filename + '.urdf')
    list_for_compstJoints = ['']
    vis_geoms_types = []

    myModel = pin.Model()
    if last_name in ['v5']:
        myJointv2 = pin.JointModelComposite(pin.JointModelPY(),pin.SE3.Identity())
        myJointv2.addJoint(pin.JointModelPZ(),pin.SE3.Identity())
        myJointv2.addJoint(pin.JointModelRX(),pin.SE3.Identity())
        myModel.addJoint(0, myJointv2, pin.SE3.Identity(), 'base')
        list_for_compstJoints.append('JointModelPlanarYZV2')
        vis_geoms_types.append(3)

    if last_name in ['v5']:
        myJointv22 = pin.JointModelComposite(pin.JointModelPY(),pin.SE3.Identity())
        myJointv22.addJoint(pin.JointModelPZ(),pin.SE3.Identity())
        myJointv22.addJoint(pin.JointModelRX(),pin.SE3.Identity())
        myModel.addJoint(0, myJointv22, pin.SE3.Identity(), 'middle_joint')
        list_for_compstJoints.append('JointModelPlanarYZV2')
        vis_geoms_types.append(3)
    for i in range(2):
        myModel.appendBodyToJoint(i+1, model_.inertias[i+1], pin.SE3.Identity())
    model = myModel
    return model, collision_model, visual_model, list_for_compstJoints, vis_geoms_types

def makeModel_grasp(problem_name):
    urdf_filename = problem_name
    model_, collision_model, visual_model = load_model_std(urdf_filename + '.urdf')
    list_for_compstJoints = ['']
    problem_name_split = problem_name.split('_')
    last_name = problem_name_split[-1]
    myModel = pin.Model()
    finger_parent = 0
    if last_name in ['v7x10']:
        offset_ = 0
        finger_parent = 1
        myJoint = pin.JointModelComposite(pin.JointModelPY(),pin.SE3.Identity())
        myJoint.addJoint(pin.JointModelPZ(),pin.SE3.Identity())
        myJoint.addJoint(pin.JointModelRX(),pin.SE3.Identity())
        joint12 = pin.JointModelPZ()
        joint12_str = 'JointModelPZ'
        joint3_str = 'JointModelPlanarYZV2'
        myModel.addJoint(0, pin.JointModelPY(), pin.SE3.Identity(), 'trans_y')
        list_for_compstJoints.append('JointModelPY')
    
    myplacement1 = pin.SE3.Identity()
    myplacement1.translation[2] = offset_
    myplacement2 = pin.SE3.Identity()
    myplacement2.translation[2] = -offset_

    myModel.addJoint(finger_parent, joint12, myplacement1, 'base')
    list_for_compstJoints.append(joint12_str)
    myModel.addJoint(finger_parent, joint12, myplacement2, 'link2')
    list_for_compstJoints.append(joint12_str)
    myModel.addJoint(0, myJoint, pin.SE3.Identity(), 'middle_joint')
    list_for_compstJoints.append(joint3_str)
    for i in range(len(list_for_compstJoints)-1):
        myModel.appendBodyToJoint(i+1, model_.inertias[i+1], pin.SE3.Identity())
    model = myModel
    return model, collision_model, visual_model, list_for_compstJoints, None

def makeModel_leaf(problem_name):
    urdf_filename = problem_name
    problem_name_split = problem_name.split('_')
    last_name = problem_name_split[-1]
    model_, collision_model, visual_model = load_model_std(urdf_filename + '.urdf')
    list_for_compstJoints = ['']

    myModel = pin.Model()
    myModel.addJoint(0, pin.JointModelPZ(), pin.SE3.Identity(), 'ball')
    list_for_compstJoints.append('JointModelPZ')
    if last_name in ['v1x10']:
        myJoint = pin.JointModelComposite(pin.JointModelPY(),pin.SE3.Identity())
        myJoint.addJoint(pin.JointModelPZ(),pin.SE3.Identity())
        myJoint.addJoint(pin.JointModelRX(),pin.SE3.Identity())
        myModel.addJoint(0, myJoint, pin.SE3.Identity(), 'coin')
        list_for_compstJoints.append('JointModelPlanarYZV2')
    if last_name in ['v1x10']:
        myModel.addJoint(0, pin.JointModelPZ(), pin.SE3.Identity(), 'floor')
        list_for_compstJoints.append('JointModelPZ')
    for i in range(len(list_for_compstJoints)-1):
        myModel.appendBodyToJoint(i+1, model_.inertias[i+1], pin.SE3.Identity())
    model = myModel
    return model, collision_model, visual_model, list_for_compstJoints