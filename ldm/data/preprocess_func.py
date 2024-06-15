from PIL import Image
import numpy as np
import cv2

def extract_lm5_from_lm68(lm68):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5 = np.stack([
        lm68[lm_idx[0], :],
        np.mean(lm68[lm_idx[[1, 2]], :], 0),
        np.mean(lm68[lm_idx[[3, 4]], :], 0), lm68[lm_idx[5], :], lm68[lm_idx[6], :]
    ],
                   axis=0)
    lm5 = lm5[[1, 2, 0, 3, 4], :]
    return lm5


def POS(xp, x):
    npts = xp.shape[0]

    A = np.zeros([2 * npts, 8])
    A[0:2 * npts - 1:2, 0:3] = x
    A[0:2 * npts - 1:2, 3] = 1
    A[1:2 * npts:2, 4:7] = x
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp, [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def resize_crop_img(ori_img, lm, trans_params):
    w0, h0, s, t0, t1, target_size = trans_params
    assert w0 == ori_img.size[0] and h0 == ori_img.size[1]

    w, h = int(w0 * s), int(h0 * s)
    left = int(w / 2 - target_size / 2 + float((t0 - w0 / 2) * s))
    right = left + target_size
    up = int(h / 2 - target_size / 2 + float((h0 / 2 - t1) * s))
    below = up + target_size

    tar_img = ori_img.resize((w, h), resample=Image.BICUBIC)
    tar_img = tar_img.crop((left, up, right, below))
    lm = np.stack([lm[:, 0] - t0 + w0/2, lm[:, 1] -
                    t1 + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])
    
    return tar_img, lm


def resize_crop_img_retain_hr(ori_img, trans_params):
    w0, h0, s, t0, t1, target_size = trans_params
    assert w0 == ori_img.size[0] and h0 == ori_img.size[1]

    w, h = int(w0 * s), int(h0 * s)
    left = int(w / 2 - target_size / 2 + float((t0 - w0 / 2) * s))
    right = left + target_size
    up = int(h / 2 - target_size / 2 + float((h0 / 2 - t1) * s))
    below = up + target_size

    sf = float(w0) / target_size
    w, h = round(w * sf), round(h * sf)
    left, up = round(left * sf), round(up * sf)
    right, below = left + w0, up + h0

    hr_img = ori_img.resize((w, h), resample=Image.BICUBIC)
    hr_img = hr_img.crop((left, up, right, below))

    return hr_img


def trans_projXY_back_to_ori_coord(projXY, trans_params):
    w0, h0, s, t0, t1, target_size = trans_params
    w, h = int(w0 * s), int(h0 * s)

    projXY = projXY + np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])
    projXY = projXY / s
    projXY_ori = np.stack(
        [np.clip(projXY[:, 0] + t0 - w0 / 2, 0, w0 - 1),
         np.clip(h0 - 1 - (projXY[:, 1] + t1 - h0 / 2), 0, h0 - 1)],
        axis=1).astype(np.float32)

    return projXY_ori
