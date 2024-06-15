import numpy as np
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from .resnet_backbone import func_dict, conv1x1, conv1x1_relu, conv3x3
from ldm.models.arcface_torch.backbones import get_model
from einops import rearrange, repeat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from inspect import isfunction
import math
from kornia.geometry import warp_affine
import cv2
def resize_n_crop(image, M, dsize=112):
    return warp_affine(image, M, dsize=(dsize, dsize))

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class ReconNetWrapper(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 use_last_fc=False,
                 fc_dim_dict=None,
                 limit_exp_range=False,
                 pretrain_model_path='./epoch_latest.pth',
                 init_SH=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 device='cuda'):
        super(ReconNetWrapper, self).__init__()

        self.use_last_fc = use_last_fc
        self.fc_dim_dict = fc_dim_dict
        self.device = device

        self.fc_dim = fc_dim_dict['id_dims'] + fc_dim_dict['exp_dims'] + fc_dim_dict['tex_dims'] + 3 + 27 + 2 + 1

        if backbone_name not in func_dict:
            return NotImplementedError('network [%s] is not implemented', backbone_name)
        func, backbone_last_dim = func_dict[backbone_name]
        self.backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)

        if not use_last_fc:
            if limit_exp_range:
                self.final_layers = nn.ModuleList([
                    conv1x1(backbone_last_dim, fc_dim_dict['id_dims'], bias=True),  
                    conv1x1_relu(backbone_last_dim, fc_dim_dict['exp_dims'], bias=True),  
                    conv1x1(backbone_last_dim, fc_dim_dict['tex_dims'], bias=True),
                    conv1x1(backbone_last_dim, 3, bias=True), 
                    conv1x1(backbone_last_dim, 27, bias=True), 
                    conv1x1(backbone_last_dim, 2, bias=True),  
                    conv1x1(backbone_last_dim, 1, bias=True)  
                ])
            else:
                self.final_layers = nn.ModuleList([
                    conv1x1(backbone_last_dim, fc_dim_dict['id_dims'], bias=True), 
                    conv1x1(backbone_last_dim, fc_dim_dict['exp_dims'], bias=True),  
                    conv1x1(backbone_last_dim, fc_dim_dict['tex_dims'], bias=True),  
                    conv1x1(backbone_last_dim, 3, bias=True),  
                    conv1x1(backbone_last_dim, 27, bias=True), 
                    conv1x1(backbone_last_dim, 2, bias=True),  
                    conv1x1(backbone_last_dim, 1, bias=True)  
                ])

            def init_weights(mm):
                if type(mm) == nn.Conv2d:
                    nn.init.constant_(mm.weight, 0.)

            for m in self.final_layers:
                try:
                    nn.init.constant_(m.weight, 0.)
                    nn.init.constant_(m.bias, 0.)
                except:
                    m.apply(init_weights)

        self.init_SH = torch.from_numpy(init_SH.reshape([1, 1, -1]).astype(np.float32)).float()

        self.load_state_dict(torch.load(pretrain_model_path, map_location=device)['net_recon'])
        print('loading the recon model from %s' % pretrain_model_path)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        coeffs_dict = self.get_coeffs(x)
        return coeffs_dict

    def get_coeffs(self, net_output):
        id_dims = self.fc_dim_dict['id_dims']
        exp_dims = self.fc_dim_dict['exp_dims']
        tex_dims = self.fc_dim_dict['tex_dims']

        id_coeffs = net_output[:, :id_dims]
        exp_coeffs = net_output[:, id_dims:id_dims + exp_dims]
        tex_coeffs = net_output[:, id_dims + exp_dims:id_dims + exp_dims + tex_dims]
        angle = net_output[:, id_dims + exp_dims + tex_dims:id_dims + exp_dims + tex_dims + 3]
        gamma = net_output[:, id_dims + exp_dims + tex_dims + 3:id_dims + exp_dims + tex_dims + 3 + 27]
        translations = net_output[:, id_dims + exp_dims + tex_dims + 3 + 27:]

        gamma = gamma.reshape(-1, 3, 9)

        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angle,
            'gamma': gamma,
            'trans': translations
        }


class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1 
        self.input_size=input_size
        
    def forward(self, image_ori):
        image = self.preprocess(F.interpolate(image_ori, size=self.input_size))
        id_feature = F.normalize(self.net(image), p=2, dim=-1)
        return id_feature


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ConditionFuser(nn.Module):
    def __init__(self): 
        super(ConditionFuser, self).__init__()
        self.inplanes = 64
        self.dilation = 1 
        self.groups = 1
        self.base_width = 64       
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        self.conv2 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.inplanes)
        self.bn4 = nn.LayerNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(BasicBlock, 128, 3, stride = 2) 

        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x)
        x = self.layer1(x) 

        x = x.flatten(2).permute(0, 2, 1) 
        x = self.bn4(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 

    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class ViTConditionFusor(nn.Module):
    def __init__(self, query_dim=192, context_dim=192, out_dim=128, heads=6, dim_head=32, dropout=0.,
                 img_size=256, lable_size=256, patch_size=16, in_chans=3, use_learnable_pos_emb=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.query_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=query_dim)
        self.context_embed = PatchEmbed(
            img_size=lable_size, patch_size=patch_size, in_chans=in_chans, embed_dim=context_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        num_query = self.query_embed.num_patches
        num_context = self.context_embed.num_patches
        
        self.query_size = self.query_embed.patch_size

        
        if use_learnable_pos_emb:
            self.pos_embed_query = nn.Parameter(torch.zeros(1, num_query, query_dim))
            self.pos_embed_context = nn.Parameter(torch.zeros(1, num_context, context_dim))
        else: 
            self.pos_embed_query = get_sinusoid_encoding_table(num_query, query_dim)
            self.pos_embed_context = get_sinusoid_encoding_table(num_context, context_dim)
            
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout),
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, context, x=None, mask=None): 
        h = self.heads
        B = x.shape[0]
        context = F.interpolate(context, (256, 256)) 
        x = self.query_embed(x)
        x = x + self.pos_embed_query.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        context = default(context, x)
        context = self.context_embed(context)
        pos_embed_context = self.pos_embed_context.expand(B, -1, -1).type_as(x).to(context.device).clone().detach()
        context = context + pos_embed_context
        
        q = self.to_q(x)
        k = self.to_k(pos_embed_context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = out.permute(0, 2, 1).reshape(B, -1, *self.query_size)
        out = self.upsample(out)
        out = out.flatten(2).permute(0, 2, 1)
        return self.to_out(out)
    
