import torch
from torch.nn.functional import one_hot

def dice(preds, labels):
    # preds, labels -> tensors
    if labels.ndim == 3:
        # need to one-hot it if labels are (N, H, W)
        labels = torch.permute(one_hot(labels.long(), num_classes=3), (0, 3, 1, 2))
    elif labels.shape == preds.shape:
        labels = labels.long()
    else:
        raise ValueError(f'Cant match shapes: {labels.shape} and {preds.shape}')

    loss = 1 - 2 * (preds * labels) / (preds.pow(2) + labels.pow(2))
    return loss.mean()