import cv2
import numpy as np

import torch
from torch.nn.functional import one_hot

# def rle_to_mask(h, w, rle, value):
#     pass


def mask_to_rle_str(mask, val):
    # np.array(mask) of shape (H, W) -> rle string for value 'val'
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    pixels = (pixels == val).astype(int)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def tensor_to_img(tensor):
    # assuming tensor has shape of [1, H, W] and is normalized to [-0.5, 0.5]
    if type(tensor) is np.ndarray:
        return tensor
    tensor = tensor.repeat(3, 1, 1)
    return ((torch.permute(tensor, (1, 2, 0)).detach().cpu().numpy() + 0.5) * 255.0).astype('uint8')


def tensor_mask_to_img(tensor):
    # assuming tensor has shape of [n_classes, H, W] for n_classes = 4 (i.e. [nothing, cl1, cl2, cl3]), so we cut the nothing-class
    if type(tensor) is np.ndarray:
        return tensor
    tensor = tensor[1:, ...]
    return ((torch.permute(tensor.round().long(), (1, 2, 0)).detach().cpu().numpy() * 255.0)).clip(0, 255).astype('uint8')


def apply_mask_to_image(img, mask, alpha=0.6):
    # assuming both img, mask are np.ndarray
    mask = mask.copy()
    mask[mask==0] = img[mask==0]
    return np.clip(img * (1-alpha) + mask * alpha, 0, 255).astype('uint8')


def save_img(tensor, fpath, mode='img'):
    # mode is either img or mask
    if mode == 'img':
        img = tensor_to_img(tensor)
    else:
        img = tensor_mask_to_img(tensor)
    cv2.imwrite(fpath, img)


def save_mask_img(img_tensor, mask_tensor, fpath):
    # img_tensor = img_tensor.clone()
    mask_tensor = mask_tensor.clone()

    img = tensor_to_img(img_tensor)
    mask = tensor_mask_to_img(mask_tensor)
    masked_img = apply_mask_to_image(img, mask)

    cv2.imwrite(fpath, masked_img)


def dice(preds, labels):
    smooth = 0.0
    eps = 1e-6
    # preds, labels -> tensors
    if labels.ndim == 3:
        # need to one-hot it if labels are (N, H, W)
        labels = torch.permute(one_hot(labels.long(), num_classes=4), (0, 3, 1, 2))
    elif labels.shape == preds.shape:
        labels = labels.long()
    else:
        raise ValueError(f'Cant match shapes: {labels.shape} and {preds.shape}')

    intersect = (preds * labels).sum(dim=[0, 2, 3])
    union = (preds + labels).sum(dim=[0, 2, 3])
    per_channel_loss = 1 - 2*(intersect + smooth)/(union + smooth + eps)
    return per_channel_loss.mean()