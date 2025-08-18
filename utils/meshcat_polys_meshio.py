import pinocchio as pin
import numpy as np
import os
from os.path import dirname, join, abspath
import meshio
import polyscope as ps

from pinocchio.visualize import MeshcatVisualizer

def load_meshes(viz:MeshcatVisualizer)->dict:
    meshes = {}
    for visual in viz.visual_model.geometryObjects:
        name = visual.name
        filename = visual.meshPath
        if name not in meshes:
            mesh = meshio.read(filename)
            meshes[name] = mesh
    return meshes

def update_and_save_to_obj(meshes_ori:dict, filename:str, viz:MeshcatVisualizer)->None:
    points = np.zeros((0, 3))
    cells = None
    num_points_accumulated = 0
    for name, mesh_ in meshes_ori.items():
        points_m = mesh_.points.copy()
        M = viz.visual_data.oMg[viz.visual_model.getGeometryId(name)]
        points_h = np.hstack([points_m, np.ones((points_m.shape[0], 1))])
        points_i = (M.homogeneous @ points_h.T).T[:, :3]
        points_k = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ points_i.T).T
        points = np.vstack([points, points_k])
        cells_now = mesh_.cells[0].data + num_points_accumulated
        if cells is None:
            cells = mesh_.cells[0].data
        else:
            cells = np.vstack([cells, cells_now])
        num_points_accumulated += mesh_.points.shape[0]
    mesh = meshio.Mesh(points, {"triangle": cells})
    meshio.write(filename, mesh)
    print('Saved to', filename)

def update_and_save_to_obj_split(meshes_ori:dict, output_dir:str, name:str, viz:MeshcatVisualizer)->None:
    
    for mesh_name, mesh_ in meshes_ori.items():
        body_dir = os.path.join(output_dir, mesh_name)
        if not os.path.exists(body_dir):
            os.makedirs(body_dir)
            
        M = viz.visual_data.oMg[viz.visual_model.getGeometryId(mesh_name)]
        
        points_m = mesh_.points.copy()
        points_h = np.hstack([points_m, np.ones((points_m.shape[0], 1))])
        points_i = (M.homogeneous @ points_h.T).T[:, :3]
        points_k = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ points_i.T).T
        # points_k = (np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ points_i.T).T
        
        mesh = meshio.Mesh(points_k, {"triangle": mesh_.cells[0].data})
        filename = os.path.join(body_dir, name)
        meshio.write(filename, mesh)
    print(f'Saved {name}')

def update_and_save_to_M_split(output_dir:str, name:str, viz:MeshcatVisualizer)->None:
    for visual in viz.visual_model.geometryObjects:
        mesh_name = visual.name
        
        body_dir = os.path.join(output_dir, mesh_name)
        if not os.path.exists(body_dir):
            os.makedirs(body_dir)
        
        M = viz.visual_data.oMg[viz.visual_model.getGeometryId(mesh_name)]
        
        matrix_file = os.path.join(body_dir, f"{name}.npy")
        np.save(matrix_file, M.homogeneous)
    
    print(f'saved M {name} to {output_dir}')