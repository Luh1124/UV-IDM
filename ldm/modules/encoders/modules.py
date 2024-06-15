import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
import kornia
import numpy as np
import cv2
from ldm.modules.x_transformer import Encoder, TransformerWrapper  
from ldm.modules.recon_network import ReconNetWrapper, ConditionFuser, ViTConditionFusor
from ldm.modules.parametric_face_model import ParametricFaceModel
from ldm.modules.seg_network import BiSeNet
from ldm.modules.renderer import get_R_T
from scipy.io import loadmat

def img3channel(img):
    '''make the img to have 3 channels'''
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def img2mask(img, thre=128, mode='greater'):
    '''mode: greater/greater-equal/less/less-equal/equal'''
    if mode == 'greater':
        mask = (img > thre).astype(np.float32)
    elif mode == 'greater-equal':
        mask = (img >= thre).astype(np.float32)
    elif mode == 'less':
        mask = (img < thre).astype(np.float32)
    elif mode == 'less-equal':
        mask = (img <= thre).astype(np.float32)
    elif mode == 'equal':
        mask = (img == thre).astype(np.float32)
    else:
        raise NotImplementedError

    mask = img3channel(mask)

    return mask


def drawlmks(img, pts):
    h, w = img.shape[:2]
    num_lmks = pts.shape[0]
    for i in range(num_lmks):
        x = int(max(min(pts[i,0],w - 1), 0))
        y = int(max(min(pts[i,1],h - 1), 0))
        img[y, x, :] = [1.0, 0.0, 0.0]
    return img
    
def perspective_projection(focal, center):
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))
    return uv_coords

def remap_tex_from_input2D(input_img, projXY, norm, seg_mask, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw):

    b, n_ver, _ = projXY.shape

    uv_ver_map_y0 = unwrap_uv_idx_v_idx[:, :, 0] 
    uv_ver_map_y1 = unwrap_uv_idx_v_idx[:, :, 1]
    uv_ver_map_y2 = unwrap_uv_idx_v_idx[:, :, 2]

    uv_XY_0 = projXY[:, uv_ver_map_y0]
    uv_XY_1 = projXY[:, uv_ver_map_y1]
    uv_XY_2 = projXY[:, uv_ver_map_y2]
    

    uv_XY = \
        uv_XY_0 * unwrap_uv_idx_bw[:, :, 0:1].unsqueeze(0) + \
        uv_XY_1 * unwrap_uv_idx_bw[:, :, 1:2].unsqueeze(0) + \
        uv_XY_2 * unwrap_uv_idx_bw[:, :, 2:3].unsqueeze(0)
    _, _, h0, w0 = input_img.shape
    uv_XY[..., 0] = (uv_XY[..., 0] / (w0 // 2)) - 1.0
    uv_XY[..., 1] = (uv_XY[..., 1] / (h0 // 2)) - 1.0
    remap_tex = F.grid_sample(input=input_img, grid=uv_XY, mode='bilinear',align_corners=False)
    remap_tex = torch.clamp(remap_tex, 0., 1.)
    
    seg_mask = seg_mask.unsqueeze(1)
    remap_seg_mask = F.grid_sample(input=seg_mask, grid=uv_XY, mode='bilinear',align_corners=False)
    ver_vis_mask = (-norm[:, :, 2] > 0.1).float() 
    remap_vis_mask0 = ver_vis_mask[:, uv_ver_map_y0] 
    remap_vis_mask1 = ver_vis_mask[:, uv_ver_map_y1]
    remap_vis_mask2 = ver_vis_mask[:, uv_ver_map_y2]
    remap_vis_mask = \
        remap_vis_mask0 * unwrap_uv_idx_bw[:, :, 0].unsqueeze(0) + \
        remap_vis_mask1 * unwrap_uv_idx_bw[:, :, 1].unsqueeze(0) + \
        remap_vis_mask2 * unwrap_uv_idx_bw[:, :, 2].unsqueeze(0)
    remap_vis_mask = (remap_vis_mask > 0.5).float().unsqueeze(1)
    remap_mask = remap_vis_mask * remap_seg_mask
    remap_tex = remap_tex * remap_mask
    return remap_tex, remap_mask


def compute_norm(vertices, face_vert_idx, vert_face_idx):

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

def trans_projXY_back_to_ori_coord(projXY, trans_params):

    device = projXY.device
    w0, h0, s, t0, t1, target_size = torch.chunk(trans_params, chunks=6, dim=1)
    w, h = (w0 * s).long(), (h0 * s).long()

    projXY = projXY + torch.cat([(w / 2 - target_size / 2), (h / 2 - target_size / 2)], dim=1).unsqueeze(1)
    projXY = projXY / s.unsqueeze(1)
    projXY_ori = torch.stack(
        [torch.clamp(projXY[..., 0] + t0 - w0 / 2, 0, w0[0].item() - 1),
         torch.clamp(h0 - 1 - (projXY[..., 1] + t1 - h0 / 2), 0, h0[0].item() - 1)],
        dim=2)

    return projXY_ori

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class Deep3dEncoder(nn.Module):
    def __init__(self,
                 pfm_model_path='BFM/BFM_model_front_5w.mat',
                 recon_model_path='checkpoints/epoch_20.pth',
                 resnet18_path='BFM/resnet_model/resnet18-5c106cde.pth',
                 maskmodel_path='BFM/parsing_model/79999_iter.pth',
                 focal=1015.0,
                 camera_distance=10.0,
                 center = 112.,
                 z_near = 5.,
                 z_far = 15.,
                 init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                 uv_coords = 'checkpoints/BFM_UV.npy',
                 device='cuda'):
        super(Deep3dEncoder, self).__init__()
        self.focal = focal
        self.camera_distance = camera_distance
        self.device = device
        init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        self.init_lit = torch.from_numpy(init_lit).to(device)
        self.facemodel = ParametricFaceModel(model_path=pfm_model_path, device=device)
        self.triangle_buffer = torch.from_numpy(np.load('BFM/triangle_buffer.npy').astype(int)).to(device)
        self.barycentric_weight = torch.from_numpy(np.load('BFM/barycentric_weight.npy')).to(device)
        self.uv_coords = torch.from_numpy(process_uv(np.load(uv_coords))).to(device)
        self.full_triangles = torch.from_numpy(loadmat('BFM/example1.mat')['full_triangles']).long().to(device)
        self.unwrap_uv_idx_v_idx = self.full_triangles[self.triangle_buffer]
        self.conditionfuser = ConditionFuser()
        self.part_idx = torch.nonzero(torch.LongTensor([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])).to(device)
        
        fc_dim_dict = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        self.persc_proj = torch.from_numpy(perspective_projection(focal, center)).to(self.device)
        self.recon_net = ReconNetWrapper(fc_dim_dict=fc_dim_dict, pretrain_model_path=recon_model_path, device=self.device)
        self.seg_model = BiSeNet(n_classes = 19, resnet18_path=resnet18_path)
        self.seg_model.load_state_dict(torch.load(maskmodel_path))
        self.seg_model.eval()
        self.part_idx = {
            'background': 0,
            'skin': 1,
            'l_brow': 2,
            'r_brow': 3,
            'l_eye': 4,
            'r_eye': 5,
            'eye_g': 6,
            'l_ear': 7,
            'r_ear': 8,
            'ear_r': 9,
            'nose': 10,
            'mouth': 11,
            'u_lip': 12,
            'l_lip': 13,
            'neck': 14,
            'neck_l': 15,
            'cloth': 16,
            'hair': 17,
            'hat': 18
        }
        self.facemodel.to(self.device)
        self.recon_net.to(self.device)
        self.recon_net.eval()

    def my_render_colors_ras_torch(self, vertices, triangles, colors, h, w, c = 3):
        ''' render mesh with colors(rasterize triangle first)
        Args:
            vertices: [nver, 3]
            triangles: [ntri, 3] 
            colors: [nver, 3]
            h: height
            w: width    
            c: channel
        Returns:
            image: [h, w, c]. rendering.
        '''
        assert vertices.shape[0] == colors.shape[1]
        device = colors.device
        triangles = triangles.to(device)
        triangle_buffer_flat = torch.reshape(self.triangle_buffer.to(device), [-1])
        barycentric_weight_flat = torch.reshape(self.barycentric_weight.to(device), [-1, c])
        weight = barycentric_weight_flat.unsqueeze(2)

        colors_flat = colors[:, triangles[triangle_buffer_flat, :], :]
        colors_flat = weight*colors_flat
        colors_flat = torch.sum(colors_flat, 2) 

        image = torch.reshape(colors_flat, [-1, h, w, c])
        image = image.permute(0, 3, 1, 2)
        return image		

    def parsing_to_mask(self, parsing):
        mask_raw = parsing[:, self.part_idx, :, :]
        return mask_raw
        
    def forward(self, x, x_ori, transM):
        with torch.no_grad():
            device = x.device
            b, c, h, w = x_ori.shape
            self.recon_net.eval()
            self.seg_model.eval()
            x_copy = x.clone()
            x_copy = (x_copy + 1) * 0.5
            x_ori = (x_ori + 1) * 0.5
            coeffs_dict = self.recon_net(x_copy)
            R, T = get_R_T(coeffs_dict['angle'], coeffs_dict['trans'])
            face_shape, _ = self.facemodel.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
            mesh_info = {'v': face_shape, 'f_v': self.facemodel.face_buf.to(device), 'v_f': self.facemodel.point_buf.to(device)}
            vertices_camera = face_shape @ R + T.unsqueeze(1)
            vertices_camera[..., -1] = self.camera_distance - vertices_camera[..., -1]
            vertices_camera = vertices_camera @ self.persc_proj.to(device)
            vertices_camera = vertices_camera[..., :2] / vertices_camera[..., 2:]
            norm = compute_norm(mesh_info['v'], mesh_info['f_v'], mesh_info['v_f'])
            norm = norm @ R
            projXY = trans_projXY_back_to_ori_coord(vertices_camera, transM)
            x_ori_resize = F.interpolate(x_ori, (512, 512))
            face_parsing_out = self.seg_model(x_ori_resize)[0]
            face_parsing_out = F.interpolate(face_parsing_out, [h, w])
            face_parsing_out = torch.argmax(face_parsing_out, dim=1)
            face_parsing_bg = (face_parsing_out == 0).float()
            face_parsing_eg = (face_parsing_out == 6).float()
            face_parsing_lear = (face_parsing_out == 7).float()
            face_parsing_rear = (face_parsing_out == 8).float()
            face_parsing_rear = (face_parsing_out == 9).float()
            face_parsing_cloth = (face_parsing_out == 16).float()
            face_parsing_hair = (face_parsing_out == 17).float()
            face_parsing_hat = (face_parsing_out == 18).float()
            seg_mask = (1 - face_parsing_bg) * (1 - face_parsing_hair) * (1 - face_parsing_eg) \
                * (1 - face_parsing_hat) * (1 - face_parsing_lear) * (1 - face_parsing_rear) * (1 - face_parsing_cloth)
            remap_tex, _ = remap_tex_from_input2D(
                x_ori,
                projXY,
                norm,
                seg_mask,
                self.unwrap_uv_idx_v_idx.to(device),
                self.barycentric_weight.to(device)
            )
            remap_tex = remap_tex*2 - 1
        c = self.conditionfuser(remap_tex)
        return c

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key

        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device) 
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):

        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):

        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.

        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):

        return self.model.encode_image(self.preprocess(x))

