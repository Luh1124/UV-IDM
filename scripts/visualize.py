import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
import cv2
import skimage
from PIL import Image
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config
from scipy.io import loadmat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.ddfa import DDFAInfer
from ldm.modules.parametric_face_model import ParametricFaceModel
from ldm.modules.renderer import get_R_T, compute_norm
from ldm.models.render_nvdiffrast import MeshRenderer
from ldm.modules.recon_network import ReconNetWrapper
from pytorch_lightning import seed_everything

def trans_back_ori_image(mask, face, ori_image, trans_params):
    w0, h0, s, t0, t1, target_size = trans_params
    w, h = int(w0 * s), int(h0 * s)
    w0, h0 = int(w0), int(h0)
    left = int(w / 2 - target_size / 2 + float((t0 - w0 / 2) * s))
    up = int(h / 2 - target_size / 2 + float((h0 / 2 - t1) * s))
    dx = left
    dy = up
    MAT = np.array([[1, 0, dx],[0, 1, dy]]).astype(np.float32)
    mask = mask[:, :, 0]
    r = face[..., 0]
    g = face[..., 1]
    b = face[..., 2]
    mask_affine = cv2.warpAffine(mask, MAT, (w, h), flags=cv2.INTER_LINEAR)
    r_affine = cv2.warpAffine(r, MAT, (w, h), flags=cv2.INTER_LINEAR)
    g_affine = cv2.warpAffine(g, MAT, (w, h), flags=cv2.INTER_LINEAR)
    b_affine = cv2.warpAffine(b, MAT, (w, h), flags=cv2.INTER_LINEAR)
    mask_affine = mask_affine[..., None]
    face_affine = np.stack([r_affine, g_affine, b_affine], axis=-1)
    mask_ori = cv2.resize(mask_affine, (w0, h0))[..., None]
    face_ori = cv2.resize(face_affine, (w0, h0))
    render_image_ori = mask_ori * face_ori + ori_image * (1 - mask_ori)
    render_image_ori = cv2.resize(render_image_ori, (224, 224))
    return render_image_ori

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy()
    triangles += 1 

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    with open(obj_name, 'w') as f:

        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                               colors[i, 1], colors[i, 2])
            f.write(s)

        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

def my_render_colors_ras_torch(vertices, triangles, colors, h, w, c = 3,triangle_buffer=None,barycentric_weight=None):
    assert vertices.shape[0] == colors.shape[1]
    


    triangle_buffer_flat = torch.reshape(triangle_buffer, [-1])
    barycentric_weight_flat = torch.reshape(barycentric_weight, [-1, c])
    weight = barycentric_weight_flat.unsqueeze(2) 

    colors_flat = colors[:, triangles[triangle_buffer_flat, :], :] 
    colors_flat = weight*colors_flat
    colors_flat = torch.sum(colors_flat, 2)

    image = torch.reshape(colors_flat, [-1, h, w, c])
    image = image.permute(0, 3, 1, 2)
    return image		

def pillow2np(img, dst_range=255.):
    coef = dst_range / 255.
    return np.asarray(img, np.float32) * coef

def read_img(path, resize=None, dst_range=255.):
    img = Image.open(path)

    if resize is not None:
        img = img.resize(resize)

    img = pillow2np(img, dst_range)

    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img

def match_color_in_yuv(src_tex, dst_tex, mask, w=1.0, sharpen=True):
    dst_tex_yuv = skimage.color.convert_colorspace(dst_tex, "rgb", "yuv")
    src_tex_yuv = skimage.color.convert_colorspace(src_tex, "rgb", "yuv")
    is_valid = mask[:, :, 0] > 0.5
    mu_dst = np.mean(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    std_dst = np.std(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    mu_src = np.mean(src_tex_yuv[is_valid], axis=0, keepdims=True)
    std_src = np.std(src_tex_yuv[is_valid], axis=0, keepdims=True)
    dst_tex_yuv_normalize = (dst_tex_yuv - mu_dst) /std_dst 
    match_tex_yuv = dst_tex_yuv_normalize * std_src + mu_src
    match_tex = skimage.color.convert_colorspace(match_tex_yuv, "yuv", "rgb")
    if sharpen:
        sharpen_op = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]], dtype=np.float32)
        match_tex = cv2.filter2D(match_tex, cv2.CV_32F, sharpen_op)
        match_tex = cv2.convertScaleAbs(match_tex, alpha=w)
    match_tex = np.clip(match_tex, 0, 255)
    return match_tex

def hair_blur(texture_map, hair_mask):
    device = texture_map.device
    texture_map_np = texture_map.clone().detach()[0].permute(1, 2, 0).cpu().numpy()
    texture_map_hair = np.zeros_like(texture_map_np)
    texture_map_hair[hair_mask == 1] = texture_map_np[hair_mask == 1]
    texture_map_hair_blur = cv2.GaussianBlur(texture_map_hair, ksize=(5,5), sigmaX=1)
    kernel = np.ones((9, 9), np.uint8)
    hair_mask = cv2.erode(hair_mask, kernel=kernel, iterations=2)
    hair_mask = cv2.blur(hair_mask, ksize=(5,5))
    texture_map = texture_map_np * (1-hair_mask) + texture_map_hair_blur * hair_mask
    texture_map = torch.from_numpy(texture_map).to(device).unsqueeze(0).permute(0, 3, 1, 2)
    return texture_map

def drawlmks(img, pts):
    h, w = img.shape[:2]
    num_lmks = pts.shape[0]
    for i in range(num_lmks):
        x = int(max(min(pts[i,0],w - 1), 0))
        y = int(max(min(pts[i,1],h - 1), 0))
        img[y, x, :] = [1.0, 0.0, 0.0]
    return img

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) 
    return uv_coords

def perspective_projection(focal, center):
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

def uv2vertices_torch(uv_map, uv_coords_part):
    uv_coords = uv_coords_part
    B, C, H ,W = uv_map.shape
    nverts = uv_coords.shape[0]
    src_x, src_y = uv_coords[:, 0], uv_coords[:, 1]
    src_x0 = torch.floor(src_x).long()
    src_x1 = src_x0 + 1
    src_y0 = torch.floor(src_y).long()
    src_y1 = src_y0 + 1

    src_x0 = torch.clip(src_x0, 0, W-1).long()
    src_x1 = torch.clip(src_x1, 0, W-1).long()
    src_y0 = torch.clip(src_y0, 0, H-1).long()
    src_y1 = torch.clip(src_y1, 0, H-1).long()

    value = torch.stack((uv_map[:, :, src_y0, src_x0],
                    uv_map[:, :, src_y0, src_x1],
                    uv_map[:, :, src_y1, src_x0],
                    uv_map[:, :, src_y1, src_x1]),dim=0).float()
    value = value.permute(1,3,2,0)

    weight = torch.stack([(src_x1 - src_x) * (src_y1 - src_y),
                    (src_x - src_x0) * (src_y1 - src_y),
                    (src_x1 - src_x) * (src_y - src_y0),
                    (src_x - src_x0) * (src_y - src_y0)],dim=0).float()
    
    weight = weight.unsqueeze(2).permute(1,0,2)
    verts = torch.matmul(value, weight)
    
    return verts[:,:,:,0]

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    epoch = pl_sd['epoch']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model, epoch

def get_input(xc, image_ori, trans_params, model):
    c = model.get_learned_conditioning(xc, image_ori, trans_params)
    return c

def trans_projXY_back_to_ori_coord(projXY, trans_params):
    w0, h0, s, t0, t1, target_size = trans_params[0]
    w, h = int(w0 * s), int(h0 * s)
    projXY = projXY.clone().detach().cpu().numpy()[0]
    projXY = projXY + np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])
    projXY = projXY / s
    projXY_ori = np.stack(
        [np.clip(projXY[:, 0] + t0 - w0 / 2, 0, w0 - 1),
         np.clip(h0 - 1 - (projXY[:, 1] + t1 - h0 / 2), 0, h0 - 1)],
        axis=1).astype(np.float32)

    return projXY_ori

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_list_file",
        type=str,
        nargs="?",
        default="",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default=""
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lmk_dir",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--show_baseline",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--show_ori",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--pfm_model_path",
        type=str,
        default='./BFM/BFM_model_front.mat',
    )
    parser.add_argument(
        "--pfm_model_path_5w",
        type=str,
        default='./BFM/BFM_model_front_5w.mat',
    )
    parser.add_argument(
        "--exp_ind_path",
        type=str,
        default='checkpoints/BFM_front_idx.mat',
    )

    parser.add_argument(
        "--uv_coords_path",
        type=str,
        default='checkpoints/BFM_UV.npy',
    )
    parser.add_argument(
        "--recon_model_path",
        type=str,
        default='checkpoints/epoch_20.pth',
    )
    parser.add_argument(
        "--draw_lmks",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--weight",
        type=str,
        default='checkpoints/ldm.ckpt',
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda device",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("configs/latent-diffusion/uvffhq-ldm-kl-4.yaml") 
    model, epoch = load_model_from_config(config, opt.weight)
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    facemodel = ParametricFaceModel(model_path=opt.pfm_model_path, device=device)
    facemodel_5w = ParametricFaceModel(model_path=opt.pfm_model_path_5w, device=device)
    triangle_buffer = torch.from_numpy(np.load('BFM/triangle_buffer.npy').astype(int)).to(device).long()
    barycentric_weight = torch.from_numpy(np.load('BFM/barycentric_weight.npy')).to(device)
    exp_ind = loadmat(opt.exp_ind_path)['idx'].astype(np.int32).squeeze() - 1
    uv_coords = torch.from_numpy(process_uv(np.load(opt.uv_coords_path))).to(device)
    uv_coords_part = uv_coords[exp_ind]
    fc_dim_dict = {
        'id_dims': facemodel.id_dims,
        'exp_dims': facemodel.exp_dims,
        'tex_dims': facemodel.tex_dims
    }
    focal = 1015.
    center = 112.
    camera_distance = 10.0
    z_near = 5.
    z_far = 15.
    full_triangles = torch.from_numpy(loadmat('BFM/example1.mat')['full_triangles']).to(device).long()
    init_lit = torch.from_numpy(np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ])).to(device)
    recon_net = ReconNetWrapper(fc_dim_dict=fc_dim_dict, pretrain_model_path=opt.recon_model_path, device=device)
    recon_net = recon_net.to(device)
    recon_net.eval()
    persc_proj = torch.from_numpy(perspective_projection(focal, center)).to(device)  
    fov = 2 * np.arctan(center / focal) * 180 / np.pi  
    renderer = MeshRenderer(
            fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)
        )
    seed_everything(opt.seed)
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    uvmappath = outpath + '_uv'
    os.makedirs(uvmappath, exist_ok=True)
    meshpath = outpath + '_mesh'
    os.makedirs(meshpath, exist_ok=True)

    dataset = DDFAInfer(
        size=224, images_list_file=opt.images_list_file, lmk_dir=opt.lmk_dir
    )
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    front_mask = read_img('masktopo/front.png', resize=(256, 256), dst_range=1.)

    all_samples=list()
    with torch.no_grad():
        for i, (xc, image_ori, trans_params, basename) in enumerate(train_loader):
            name = basename[0]+'.png'
            xc = xc.permute(0, 3, 1, 2).to(model.device)
            image_ori = image_ori.permute(0, 3, 1, 2).to(model.device)
            trans_params = trans_params.float().to(model.device)
            c = get_input(xc, image_ori, trans_params, model)
            samples, _ = model.sample_log(
                cond=c, batch_size=1, ddim=True, ddim_steps=opt.ddim_steps, eta=opt.ddim_eta
            )
            x_sample = model.decode_first_stage(samples.to(model.device))
            xc = (xc + 1.0) * 0.5 
            x_sample = (x_sample + 1.0) * 0.5

            coeffs_dict = recon_net(xc)
            xc = xc.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
            R, T = get_R_T(coeffs_dict['angle'], coeffs_dict['trans'])
            face_shape, _ = facemodel.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
            vertices_camera = face_shape @ R + T.unsqueeze(1)
            vertices_camera_v2 = face_shape + T.unsqueeze(1)
            vertices_camera[..., -1] = camera_distance - vertices_camera[..., -1]
            vertices_camera_v2[..., -1] = camera_distance - vertices_camera_v2[..., -1]
            if opt.show_baseline:
                face_texture = facemodel.compute_texture(coeffs_dict['tex'])
                norm = compute_norm(face_shape, facemodel.face_buf_part.to(device), facemodel.point_buf.to(device) - 1)
                norm = norm @ R 
                face_color = facemodel.compute_color(face_texture, norm, init_lit, coeffs_dict['gamma'])
 
            face_texture = facemodel_5w.compute_texture(coeffs_dict['tex'])
            face_texture[:, exp_ind] = face_color
            face_color_map = my_render_colors_ras_torch(uv_coords, full_triangles, face_texture, 256, 256, 3, triangle_buffer, barycentric_weight)
            
            face_color_map = face_color_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            x_sample = x_sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            face_color_map = np.clip(face_color_map, 0., 1.)
            x_sample = np.clip(x_sample, 0., 1.)
            Image.fromarray((x_sample*255.).astype(np.uint8)).save(os.path.join(uvmappath, name))
            x_sample = match_color_in_yuv(face_color_map*255., x_sample*255., front_mask, sharpen=False)/255.
            x_sample = torch.from_numpy(x_sample).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            
            x_texture = uv2vertices_torch(x_sample, uv_coords_part)
            mask, _ , face = renderer(vertices_camera, facemodel.face_buf_part.to(device), x_texture)
            
            face = face.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
            mask = mask.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
            trans_params = trans_params.clone().detach().cpu().numpy()[0]
            image_ori = image_ori.clone().detach().cpu().numpy()[0].transpose(1, 2, 0)
            image_ori = (image_ori + 1) * 0.5
            x_render = trans_back_ori_image(mask, face, image_ori, trans_params)
            x_render = np.clip(x_render, 0, 1.)
            Image.fromarray((x_render*255.).astype(np.uint8)).save(os.path.join(outpath, name))
            write_obj_with_colors(os.path.join(meshpath, name), vertices_camera_v2.clone().detach().cpu().numpy()[0], 
                                  facemodel.face_buf_part.clone().detach().cpu().numpy(), x_texture.clone().detach().cpu().numpy()[0])


    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
