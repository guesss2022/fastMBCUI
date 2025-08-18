import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from os.path import dirname, join, abspath

def load_model_std(urdf_filename):
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "example-robot-data/robots")
    mesh_dir = pinocchio_model_dir
    urdf_model_path = join(join(model_path, "guesss_description/robots"), urdf_filename)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )
    return model, collision_model, visual_model

def get_qr_from_rotvec(rotvec, model):
    rotquat = R.from_rotvec(rotvec).as_quat()
    qr = pin.neutral(model)
    qr[3:7] = rotquat
    return qr