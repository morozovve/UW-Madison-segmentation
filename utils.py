import torch
from torch.nn.functional import one_hot

def dice(preds, labels):
    smooth = 0.0
    eps = 1e-6
    # preds, labels -> tensors
    if labels.ndim == 3:
        # need to one-hot it if labels are (N, H, W)
        labels = torch.permute(one_hot(labels.long(), num_classes=3), (0, 3, 1, 2))
    elif labels.shape == preds.shape:
        labels = labels.long()
    else:
        raise ValueError(f'Cant match shapes: {labels.shape} and {preds.shape}')

    intersect = (preds * labels).sum(dim=[0, 2, 3])
    union = (preds + labels).sum(dim=[0, 2, 3])
    per_channel_loss = 1 - 2*(intersect + smooth)/(union + smooth + eps)
    return per_channel_loss.mean()