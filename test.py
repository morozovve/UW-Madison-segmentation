import torch
from torch.nn.functional import one_hot

torch.manual_seed(0)

from utils import dice
from model import UNet

def dice_test():
    print('Testing dice')
    # non-matching shapes
    lbls = (torch.rand((2, 32, 32)) * 3).long()
    lbls_oh = torch.permute(one_hot(lbls, num_classes=3), (0, 3, 1, 2))
    preds = torch.rand((2, 3, 32, 32))
    res1 = dice(preds, lbls)
    
    # matching shapes
    res2 = dice(preds, lbls_oh)

    assert res1 == res2, 'res1 != res2'
    print(f'res = {res1}')
    print('Done')

def unet_test():
    print('Testing UNet')
    n_cl = 3
    inp = torch.rand((2,3,256,256)) - 0.5
    m = UNet(n_classes=n_cl)
    res = m.forward(inp)
    assert res.shape[2:] == inp.shape[2:], f'shapes different in w/h dim: {res.shape[2:]} and {inp.shape[2:]}'
    assert res.shape[1] == n_cl, f'ch dim {res.shape[1]} != n_classes {n_cl}'
    print('Done')

if __name__ == '__main__':
    dice_test()
    unet_test()