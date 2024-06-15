import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
from typing import List
from scipy.io import loadmat


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]

class ParametricFaceModel(nn.Module):

    def __init__(self, model_path, device='cuda'):
        super(ParametricFaceModel, self).__init__()
        model = loadmat(model_path)


        self.mean_shape = model['meanshape'].astype(np.float32)

        self.mean_tex = model['meantex'].astype(np.float32)
        mean_shape = self.mean_shape.reshape([-1, 3])
        mean_shape = mean_shape - np.array([-0.003226267, 0.045060683, 0.75983167])[np.newaxis,:].astype(np.float32)
        self.mean_shape = mean_shape.reshape([1, -1])

        self.id_base = model['idBase'].astype(np.float32)
 
        self.exp_base = model['exBase'].astype(np.float32)

        self.tex_base = model['texBase'].astype(np.float32)

        self.id_dims = self.id_base.shape[1]
        self.exp_dims = self.exp_base.shape[1]
        self.tex_dims = self.tex_base.shape[1]



        self.face_buf = loadmat('BFM/example1.mat')['full_triangles'].astype(int)
        self.face_buf_part = model['tri'].astype(np.int64) - 1

        self.point_buf = model['point_buf'].astype(np.int64) 

        self.SH = SH()
        self.np2tensor()
        self.face_buf.to(device)
    def np2tensor(self):
        '''
        Transfer numpy.array to torch.Tensor.
        '''
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value))

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == torch.__name__:
                setattr(self, key, value.to(device))

    def compute_shape(self, id_coeff, exp_coeff):

        batch_size = id_coeff.shape[0]
        device = id_coeff.device
        id_part = torch.einsum('ij,aj->ai', self.id_base.to(device), id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base.to(device), exp_coeff)
        face_shape = (id_part + exp_part + self.mean_shape.to(device)).reshape([batch_size, -1, 3])
        id_shape = (id_part + self.mean_shape.to(device)).reshape([batch_size, -1, 3])
        return face_shape, id_shape

    def compute_texture(self, tex_coeff, normalize=True):

        batch_size = tex_coeff.shape[0]
        device = tex_coeff.device
        tex_part = torch.einsum('ij,aj->ai', self.tex_base.to(device), tex_coeff)
        face_texture = (tex_part + self.mean_tex.to(device)).reshape([batch_size, -1, 3])
        if normalize:
            face_texture = face_texture / 255.
        return face_texture

    def compute_color(self, face_texture, face_norm, init_lit, gamma):

            batch_size = gamma.shape[0]
            v_num = face_texture.shape[1]
            a, c = self.SH.a, self.SH.c
            gamma = gamma.reshape([batch_size, 3, 9])
            gamma = gamma + init_lit.float()
            gamma = gamma.permute(0, 2, 1)
            Y = torch.cat([
                a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(gamma.device),
                -a[1] * c[1] * face_norm[..., 1:2],
                a[1] * c[1] * face_norm[..., 2:],
                -a[1] * c[1] * face_norm[..., :1],
                a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
                -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
                0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
                -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
                0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
            ], dim=-1)
            r = Y @ gamma[..., :1]
            g = Y @ gamma[..., 1:2]
            b = Y @ gamma[..., 2:]
            face_color = torch.cat([r, g, b], dim=-1) * face_texture
            return face_color