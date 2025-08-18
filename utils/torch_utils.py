import torch
import numpy as np

def compute_normal_vertex(V, F):
    """
    Compute the vertex normals for a mesh using face normals and angle weights (vectorized version).
    
    Parameters:
        V (numpy.ndarray): Array of vertices of shape (N, 3), where N is the number of vertices.
        F (numpy.ndarray): Array of faces of shape (M, 3), where M is the number of faces.

    Returns:
        normal_vertex (numpy.ndarray): Array of vertex normals of shape (N, 3).
    """
    # Step 1: Compute edge vectors
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p2 - p1

    # Step 2: Compute face normals
    face_normals = np.cross(e1, e2)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Step 3: Compute angles at each vertex of the face
    def compute_angle(a, b):
        """Helper function to compute the angle between two vectors."""
        dot_product = np.einsum('ij,ij->i', a, b)
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
    
    angle0 = compute_angle(e1, e2)
    angle1 = compute_angle(-e1, e3)
    angle2 = np.pi - angle0 - angle1

    # Step 4: Accumulate angle-weighted normals for each vertex
    vertex_contributions = np.zeros_like(V)
    for i, angles in enumerate([angle0, angle1, angle2]):
        np.add.at(vertex_contributions, F[:, i], face_normals * angles[:, np.newaxis])

    # Step 5: Normalize vertex normals
    normal_vertex = vertex_contributions / np.linalg.norm(vertex_contributions, axis=1, keepdims=True)

    return normal_vertex

def compute_face_centers(V, F):
    ind0 = F[:, 0]
    ind1 = F[:, 1]
    ind2 = F[:, 2]

    vertices_0 = V[ind0]
    vertices_1 = V[ind1]
    vertices_2 = V[ind2]

    C = (vertices_0 + vertices_1 + vertices_2) / 3.0
    
    return C

def compute_FL_matrix(V, F, Cr):
    ind0, ind1, ind2 = F[:, 0], F[:, 1], F[:, 2]
    vertices_0, vertices_1, vertices_2 = V[ind0], V[ind1], V[ind2]

    edge1 = vertices_1 - vertices_0
    edge2 = vertices_2 - vertices_0
    
    normal = torch.cross(edge1, edge2, dim=1) / 2.0
    
    cross_center_normal = torch.cross(Cr, normal, dim=1)

    FL = torch.cat((cross_center_normal, normal), dim=1)

    return FL

def compute_FL_matrix_onC_noMulArea(V, F, Cr):
    ind0, ind1, ind2 = F[:, 0], F[:, 1], F[:, 2]
    vertices_0, vertices_1, vertices_2 = V[ind0], V[ind1], V[ind2]

    edge1 = vertices_1 - vertices_0
    edge2 = vertices_2 - vertices_0
    
    normal = torch.cross(edge1, edge2, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True)  # Normalize normal vectors
    
    cross_center_normal = torch.cross(Cr, normal, dim=1)

    FL = torch.cat((cross_center_normal, normal), dim=1)

    return FL

def compute_FL_matrix_onV_noMulArea(Vr, normal):
    cross_center_normal = torch.cross(Vr, normal, dim=1)

    FL = torch.cat((cross_center_normal, normal), dim=1)

    return FL

def laplace3d_dlnmk_matrix_torch(targets, sources, n_targets, n_sources, p=1e-4):
    eps2 = p ** 2

    r = targets[:, torch.newaxis, :] - sources[torch.newaxis, :, :]  # (m,n,3)
    
    reg_r2 = torch.sum(r**2, axis=2) + eps2  # (m,n)
    reg_r1 = torch.sqrt(reg_r2)
    reg_r3 = reg_r1 * reg_r2
    
    rnx = torch.sum(r * n_targets[:, torch.newaxis, :], axis=2)   # (m,n)
    rny = torch.sum(r * n_sources[torch.newaxis, :, :], axis=2)
    nxny = torch.einsum('ik,jk->ij', n_targets, n_sources)     # (m,n)
    
    R_4PI = 0.07957747154594767
    gra = R_4PI / reg_r1
    
    term1 = gra / reg_r2 * nxny
    term2 = 3 * gra * rnx * rny / (reg_r1 * reg_r3)
    kernel_matrix = term1 - term2
    
    return kernel_matrix

def laplace3d_slnmk_matrix_torch(V, F, S, C):
    ind0, ind1, ind2 = F[:, 0], F[:, 1], F[:, 2]
    vertices_0, vertices_1, vertices_2 = V[ind0], V[ind1], V[ind2]

    edge1 = vertices_1 - vertices_0
    edge2 = vertices_2 - vertices_0
    
    normal = torch.cross(edge1, edge2, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True)  # Normalize normal vectors

    r = C[:, torch.newaxis, :] - S[torch.newaxis, :, :]  # (m,n,3)
    
    reg_r2 = torch.sum(r**2, axis=2)  # (m,n)
    reg_r1 = torch.sqrt(reg_r2)
    reg_r3 = reg_r1 * reg_r2
    
    rnx = torch.sum(r * normal[:, torch.newaxis, :], axis=2)   # (m,n)
    
    kernel_matrix = 1.0 / reg_r3 * rnx
    
    return kernel_matrix

def compute_aggregate_ILdrag(V, F, Cr):
    v0 = V[F[:, 0]]  # (M,3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = torch.cross(e1, e2, dim=1)
    areas = torch.norm(cross, dim=1) / 2  # (M,)
    
    M = F.shape[0]
    R = torch.zeros((M, 3, 3), dtype=V.dtype, device=V.device)
    r = Cr  # (M, 3)
    R[:, 0, 1] = -r[:, 2]
    R[:, 0, 2] = r[:, 1]
    R[:, 1, 0] = r[:, 2]
    R[:, 1, 2] = -r[:, 0]
    R[:, 2, 0] = -r[:, 1]
    R[:, 2, 1] = r[:, 0]
    
    RR = torch.bmm(R, R)
    
    weighted_RR = RR * areas.view(-1, 1, 1)
    sum_RR_area = weighted_RR.sum(dim=0)
    
    weighted_R = R * areas.view(-1, 1, 1)
    sum_R_area = weighted_R.sum(dim=0)
    
    sum_area = areas.sum()
    
    ILdrag = torch.zeros((6, 6), dtype=V.dtype, device=V.device)
    ILdrag[:3, :3] = sum_RR_area
    ILdrag[:3, 3:6] = -sum_R_area
    ILdrag[3:6, :3] = sum_R_area
    ILdrag[3:6, 3:6] = -sum_area * torch.eye(3, dtype=V.dtype, device=V.device)
    
    return ILdrag

def compute_aggregate_ILTdrag(V, F, Cr):
    v0 = V[F[:, 0]]  # (M,3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = torch.cross(e1, e2, dim=1)
    areas = torch.norm(cross, dim=1) / 2  # (M,)
    
    M = F.shape[0]
    R = torch.zeros((M, 3, 3), dtype=V.dtype, device=V.device)
    r = Cr  # (M, 3)
    R[:, 0, 1] = -r[:, 2]
    R[:, 0, 2] = r[:, 1]
    R[:, 1, 0] = r[:, 2]
    R[:, 1, 2] = -r[:, 0]
    R[:, 2, 0] = -r[:, 1]
    R[:, 2, 1] = r[:, 0]
    
    norm = torch.norm(cross, dim=1, keepdim=True)
    n = cross / (norm + 1e-8)
    outer_n = torch.einsum('bi,bj->bij', n, n)
    eye = torch.eye(3, dtype=V.dtype, device=V.device).unsqueeze(0)
    P = eye - outer_n
    
    Mx3x6 = torch.zeros((M, 3, 6), dtype=V.dtype, device=V.device)
    Mx3x6[:, :, :3] = -R
    Mx3x6[:, :, 3:] = torch.eye(3, dtype=V.dtype, device=V.device)
    Mx6x3 = Mx3x6.transpose(1, 2).clone()

    Mx3x6_p = torch.einsum('bij,bjk->bik', P, Mx3x6)
    Mx6x6 = torch.einsum('bij,bjk->bik', -Mx6x3, Mx3x6_p)
    Mx6x6 = Mx6x6 * areas.view(-1, 1, 1)
    ILdrag = Mx6x6.sum(dim=0)
    
    return ILdrag

def compute_aggregate_ILNdrag(V, F, Cr):
    v0 = V[F[:, 0]]  # (M,3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = torch.cross(e1, e2, dim=1)
    areas = torch.norm(cross, dim=1) / 2  # (M,)
    
    M = F.shape[0]
    R = torch.zeros((M, 3, 3), dtype=V.dtype, device=V.device)
    r = Cr  # (M, 3)
    R[:, 0, 1] = -r[:, 2]
    R[:, 0, 2] = r[:, 1]
    R[:, 1, 0] = r[:, 2]
    R[:, 1, 2] = -r[:, 0]
    R[:, 2, 0] = -r[:, 1]
    R[:, 2, 1] = r[:, 0]
    
    norm = torch.norm(cross, dim=1, keepdim=True)
    n = cross / (norm + 1e-8)
    P = torch.einsum('bi,bj->bij', n, n)
    
    Mx3x6 = torch.zeros((M, 3, 6), dtype=V.dtype, device=V.device)
    Mx3x6[:, :, :3] = -R
    Mx3x6[:, :, 3:] = torch.eye(3, dtype=V.dtype, device=V.device)
    Mx6x3 = Mx3x6.transpose(1, 2).clone()

    Mx3x6_p = torch.einsum('bij,bjk->bik', P, Mx3x6)
    Mx6x6 = torch.einsum('bij,bjk->bik', -Mx6x3, Mx3x6_p)
    Mx6x6 = Mx6x6 * areas.view(-1, 1, 1)
    ILdrag = Mx6x6.sum(dim=0)
    
    return ILdrag


def compute_single_layer(S, C):
    '''
    S: tensor of shape (num_sources, 3)
    C: tensor of shape (num_faces, 3)
    Returns a tensor SL of shape (num_faces, num_sources)
    '''
    # Expand and difference the dimensions to broadcast for subtraction
    # C: (num_faces, 1, 3) -> (num_faces, num_sources, 3)
    C_expanded = C.unsqueeze(1)

    # S: (1, num_sources, 3) -> (num_faces, num_sources, 3)
    S_expanded = S.unsqueeze(0)

    # Calculate difference
    diff = C_expanded - S_expanded  # Shape is (num_faces, num_sources, 3)

    # Calculate norms
    norms = torch.norm(diff, dim=2)  # Shape is (num_faces, num_sources)

    # Compute single layer kernel matrix
    SL = 1.0 / norms
    
    return SL

def compute_double_layer(S, C, N, p):
    '''
    S: tensor of shape (num_sources, 3)
    C: tensor of shape (num_faces, 3)
    N: tensor of shape (num_sources, 3) normals at source points
    Returns a tensor DL of shape (num_faces, num_sources)
    '''
    
    # Expand and difference the dimensions to broadcast for subtraction
    # C: (num_faces, 1, 3) -> (num_faces, num_sources, 3)
    C_expanded = C.unsqueeze(1)

    # S: (1, num_sources, 3) -> (num_faces, num_sources, 3)
    S_expanded = S.unsqueeze(0)

    # Calculate difference
    diff = C_expanded - S_expanded  # Shape is (num_faces, num_sources, 3)

    # Calculate norms
    norms2 = torch.sum(diff**2, dim=2) + p**2  # Shape is (num_faces, num_sources)
    norms = torch.sqrt(norms2)  # Shape is (num_faces, num_sources)

    # Compute double layer kernel matrix
    R_4PI = 0.07957747154594767
    DL = R_4PI / norms / norms / norms * (N[torch.newaxis, :, :] * diff).sum(dim=-1)  # Shape is (num_faces, num_sources)
    
    return DL

def compute_solid_angle(V, F, S):

    R1 = V[F[:, 0]]
    R2 = V[F[:, 1]]
    R3 = V[F[:, 2]]

    for_s_expanded = S.unsqueeze(1)
    R1_minus_s = R1 - for_s_expanded
    R2_minus_s = R2 - for_s_expanded
    R3_minus_s = R3 - for_s_expanded

    cross_R2_R3 = torch.cross(R2_minus_s, R3_minus_s, dim=2)

    numerator = torch.einsum('ijk,ijk->ij', cross_R2_R3, R1_minus_s)

    l1, l2, l3 = (
        torch.norm(R1_minus_s, dim=2),
        torch.norm(R2_minus_s, dim=2),
        torch.norm(R3_minus_s, dim=2)
    )

    dot_R1_R2 = torch.einsum('ijk,ijk->ij', R1_minus_s, R2_minus_s)
    dot_R1_R3 = torch.einsum('ijk,ijk->ij', R1_minus_s, R3_minus_s)
    dot_R2_R3 = torch.einsum('ijk,ijk->ij', R2_minus_s, R3_minus_s)
    den = l1 * l2 * l3 + dot_R1_R2 * l3 + dot_R1_R3 * l2 + dot_R2_R3 * l1

    solid_angles = 2 * torch.atan2(numerator, den).transpose(0, 1)

    return solid_angles

def compute_quadrature(F, V, C):
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    
    s = torch.cross(p1 - p0, p2 - p0, dim=1) / 2
    
    rs = torch.cross(C, s, dim=1)
    
    Q = torch.cat((rs, s), dim=1).T
    
    return Q

def compute_quat2matrix(quat):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
        quat (torch.Tensor): A tensor of shape (4,) representing a quaternion.

    Returns:
        R (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
    """
    # Extract the components of the quaternion
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    
    # Compute the rotation matrix
    R = torch.tensor([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]
    ], dtype=quat.dtype, device=quat.device)
    
    return R

def compute_surf_area(V, F):
    R1 = V[F[:, 0]]
    R2 = V[F[:, 1]]
    R3 = V[F[:, 2]]

    normal = torch.cross(R2 - R1, R3 - R1, dim=1)

    area = 0.5 * torch.norm(normal, dim=1)
    
    return area

def compute_EField_value(VS, source, C):
    VS = VS.unsqueeze(0)  # (1, Ns, 3)
    C = C.unsqueeze(1)    # (Nf, 1, 3)
    source = source.unsqueeze(0).unsqueeze(-1)  # (1, Ns, 1)
    
    r = C - VS  # (Nf, Ns, 3)
    r_norm = torch.norm(r, dim=2, keepdim=True)  # (Nf, Ns, 1)
    
    E = (source) * (r / (r_norm ** 3))  # (Nf, Ns, 3)
    
    E_total = torch.sum(E, dim=1)  # (Nf, 3)
    
    return E_total

def compute_potential_integral(VS, source, C, dS):
    VS = VS.unsqueeze(0)  # (1, Ns, 3)
    C = C.unsqueeze(1)    # (Nf, 1, 3)
    source = source.unsqueeze(0).unsqueeze(-1)  # (1, Ns, 1)

    r = C - VS  # (Nf, Ns, 3)
    r_norm = torch.norm(r, dim=2, keepdim=True)  # (Nf, Ns, 1)

    phi = source / r_norm  # (Nf, Ns, 1)

    phi_total = torch.sum(phi, dim=1)  # (Nf, 1)

    phi_dS = phi_total * dS  # (Nf, 3)

    integral_result = torch.sum(phi_dS, dim=0)  # (3,)

    return integral_result

def compute_EFieldSquared_integral(VS, source, C, dS):
    VS = VS.unsqueeze(0)  # (1, Ns, 3)
    C = C.unsqueeze(1)    # (Nf, 1, 3)
    source = source.unsqueeze(0).unsqueeze(-1)  # (1, Ns, 1)

    r = C - VS  # (Nf, Ns, 3)
    r_norm = torch.norm(r, dim=2, keepdim=True)  # (Nf, Ns, 1)

    E = (source) * (r / (r_norm ** 3))  # (Nf, Ns, 3)

    E_squared = torch.norm(E, dim=2, keepdim=True) ** 2  # (Nf, Ns, 1)

    E_squared_total = torch.sum(E_squared, dim=1)  # (Nf, 1)

    E_squared_dS = E_squared_total * dS  # (Nf, 3)

    integral_result = torch.sum(E_squared_dS, dim=0)  # (3,)

    return integral_result

def compute_dS(V, F):
    R1 = V[F[:, 0]]
    R2 = V[F[:, 1]]
    R3 = V[F[:, 2]]

    normal = torch.cross(R2 - R1, R3 - R1, dim=1)

    dS = 0.5 * normal

    return dS