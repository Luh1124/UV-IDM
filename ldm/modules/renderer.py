import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
from typing import List
from scipy.io import loadmat



def get_R_T(angle, trans):
    '''
    Get rotation and translation matrices (internal camera parameters).

    Args:
        angle: torch.Tensor, (B, 3). The estimated angle.
        trans: torch.Tensor, (B, 3). The estimated trans.
    Returns:
        R: torch.Tensor, (B, 3, 3). The rotation matrix.
        T: torch.Tensor, (B, 3). The translation matrix.
    '''
    batch_size = angle.shape[0]
    ones = torch.ones([batch_size, 1]).type_as(angle)
    zeros = torch.zeros([batch_size, 1]).type_as(angle)
    x, y, z = angle[:, :1], angle[:, 1:2], angle[:, 2:]

    rot_x = torch.cat(
        [ones, zeros, zeros, zeros,
         torch.cos(x), -torch.sin(x), zeros,
         torch.sin(x), torch.cos(x)], dim=1).reshape([batch_size, 3, 3])
    rot_y = torch.cat(
        [torch.cos(y), zeros, torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
         torch.cos(y)], dim=1).reshape([batch_size, 3, 3])
    rot_z = torch.cat(
        [torch.cos(z), -torch.sin(z), zeros,
         torch.sin(z), torch.cos(z), zeros, zeros, zeros, ones], dim=1).reshape([batch_size, 3, 3])
    R = (rot_z @ rot_y @ rot_x).permute(0, 2, 1).type_as(angle)
    T = trans.clone().type_as(trans)

    return R, T


def get_pin_hole_projection_matrix(focal, center):
    '''
    Get projection matrix of pin hole perspective camera (internal camera parameters).

    Args:
        focal: int. The camera focal length.
        center: int. The imaging center.
    Returns:
        proj_mat: torch.Tensor, (3, 3). The projection matrix.
    '''
    proj_mat = np.array([focal, 0, center, 0, focal, center, 0, 0, 1], dtype=np.float32).reshape([3, 3]).transpose()
    proj_mat = torch.from_numpy(proj_mat).float()
    return proj_mat




def world_to_camera_transform(vertices_world, R, T, camera_distance):
    '''
    Tranform the world coordinates to the camera coordinates according to R, T, and camera_distance.

    Args:
        vertices_world: torch.Tensor, (B, N, 3). The input vertices in the world coordinate system.
        R: torch.Tensor, (B, 3, 3). The rotation matrix.
        T: torch.Tensor, (B, 3). The translation matrix.
        camera_distance: float. The camera distance.
    Returns:
        vertices_camera: torch.Tensor, (B, N, 3). The output vertices in the camera coordinate system.
    '''
    vertices_camera = vertices_world @ R + T.unsqueeze(1)
    vertices_camera[..., -1] = camera_distance - vertices_camera[..., -1]
    return vertices_camera


def camera_to_imaging_transform(vertices_camera, proj_mat):
    '''
    Tranform the camera coordinates to the imaging coordinates according to projection matrix.

    Args:
        vertices_camera: torch.Tensor, (B, N, 3). The input vertices in the camera coordinate system.
        proj_mat: torch.Tensor, (3, 3). The perspective projection matrix.
    Returns:
        vertices_imaging: torch.Tensor, (B, N, 2). The output vertices in the imaging coordinate system.
    '''
    vertices_imaging = vertices_camera @ proj_mat
    vertices_imaging = vertices_imaging[..., :2] / vertices_imaging[..., 2:]
    return vertices_imaging




def compute_norm(vertices, face_vert_idx, vert_face_idx):
    '''
    Compute the normal vector according the shape.

    Args:
        vertices: torch.Tensor, (B, N, 3). The vertices (world coordinate system).
        face_vert_idx: torch.Tensor, (M, 3). The vertex indices for each face.
        vert_face_idx: torch.Tensor, (N, 8). The face indices for each vertex that lies in.
    Returns:
        vertex_norm: torch.Tensor, (B, N, 3). The normal vector for each vertex.
    '''
    v1 = vertices[:, face_vert_idx[:, 0]]  
    v2 = vertices[:, face_vert_idx[:, 1]]  
    v3 = vertices[:, face_vert_idx[:, 2]] 

    e1 = v1 - v2 
    e2 = v2 - v3 
    face_norm = torch.cross(e1, e2, dim=-1)
    face_norm = F.normalize(face_norm, dim=-1, p=2)
    face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).type_as(face_norm)], dim=1)

    vertex_norm = face_norm[:, vert_face_idx] 
    vertex_norm = torch.sum(vertex_norm, dim=2)  
    vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)  
    return vertex_norm



def compute_224projXY_norm_by_pin_hole(mesh_info, angle, trans, camera_distance, focal):
    '''
    Compute project XY coordinates (224x224) and normal vectors for each vertex by pin hole camera.

    Args:
        mesh_info: dict. The information of the mesh, should contain:
            'v': torch.Tensor, (B, N, 3). The coordinates of vertices (world coordinate system).
            'f_v': torch.Tensor, (M, 3). The vertex indices for each face.
            'v_f': torch.Tensor, (N, 8). The face indices for each vertex that lies in.
        angle: torch.Tensor, (B, 3). The estimated angle.
        trans: torch.Tensor, (B, 3). The estimated trans.
        camera_distance: float. The camera distance.
        focal: int. The camera focal length.
    Returns:
        projXY: torch.Tensor, (B, N, 2). The project XY coordinates (224x224) for each vertex.
        norm: torch.Tensor, (B, N, 3). The normal vector for each vertex.
    '''


    R, T = get_R_T(angle, trans)
    vertices_camera = world_to_camera_transform(mesh_info['v'], R, T, camera_distance)


    proj_mat = get_pin_hole_projection_matrix(focal=focal, center=112.)
    proj_mat = proj_mat.type_as(vertices_camera)
    projXY = camera_to_imaging_transform(vertices_camera, proj_mat)


    norm = compute_norm(mesh_info['v'], mesh_info['f_v'], mesh_info['v_f'])  
    norm = norm @ R

    return projXY, norm

    


class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        super(MeshRenderer, self).__init__()

        self.rasterize_size = rasterize_size
        self.fov = rasterize_fov
        self.znear = znear
        self.zfar = zfar

        self.rasterizer = None
    
    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N ,C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]



        if self.rasterizer is None:
            self.rasterizer = MeshRasterizer()
            print("create rasterizer on device cuda:%d"%device.index)
        


        tri = tri.type(torch.int32).contiguous()

        cameras = FoVPerspectiveCameras(
            device=device,
            fov=self.fov,
            znear=self.znear,
            zfar=self.zfar,
        )

        raster_settings = RasterizationSettings(
            image_size=rsize
        )


        mesh = Meshes(vertex.contiguous()[...,:3], tri.unsqueeze(0).repeat(vertex.shape[0], 1, 1))

        fragments = self.rasterizer(mesh, cameras = cameras, raster_settings = raster_settings)
        rast_out = fragments.pix_to_face.squeeze(-1)
        depth = fragments.zbuf


        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if feat is not None:
            attributes = feat.reshape(-1,3)[mesh.faces_packed()]
            image = pytorch3d.ops.interpolate_face_attributes(fragments.pix_to_face,
                                                      fragments.bary_coords,
                                                      attributes)

            image = image.squeeze(-2).permute(0, 3, 1, 2)
            image = mask * image
        
        return mask, depth, image