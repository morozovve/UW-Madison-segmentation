import torch
import cv2
import numpy as np
from torch.utils import data
from tqdm import tqdm

def split_train_test(keys, ratio=.8):
    case_days = sorted(set('_'.join(x.split('_')[:2]) for x in keys))
    ntrain = int(len(case_days) * ratio)
    train_case_days, test_case_days = case_days[:ntrain], case_days[ntrain:]
    train_keys = [x for x in keys if '_'.join(x.split('_')[:2]) in train_case_days]
    test_keys = [x for x in keys if '_'.join(x.split('_')[:2]) in test_case_days]
    return train_keys, test_keys

def get_resize_both(size=256):
    def resize_both(x, y):
        x = cv2.resize(x, (size, size))
        y = cv2.resize(y, (size, size))
        return x, y
    return resize_both


def get_to_tensor_both(xtype=torch.float32, ytype=torch.long):
    def to_tensor_both(x, y):
        x = x[np.newaxis, ...]
        return torch.tensor(x, dtype=xtype), torch.tensor(y, dtype=ytype)
    return to_tensor_both


def normalize(x, y):
    x = (x - x.min()) / (x.max() - x.min()) - 0.5
    return x, y


def get_chain_transforms(*args):
    def chain_fn(x, y):
        for func in args:
            x, y = func(x, y)
        return x, y
    return chain_fn


def load_train_data(df):
    keys = sorted(df.keys())

    train_keys, test_keys = split_train_test(keys, ratio=.8)

    transform = get_chain_transforms(
        get_resize_both(size=256),
        get_to_tensor_both(),
        normalize,
    )

    ds_train = SegmentationDataSet({k: df[k] for k in train_keys}, transform=transform)
    ds_test  = SegmentationDataSet({k: df[k] for k in test_keys},  transform=transform)

    dl_train = data.DataLoader(dataset=ds_train,
                               batch_size=8,
                               shuffle=True)
    dl_test  = data.DataLoader(dataset=ds_test,
                               batch_size=8,
                               shuffle=False)
    return dl_train, dl_test


# Creating dataset
class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 data_dict: dict,
                 transform=None,
                 ):

        self.inputs = sorted(data_dict.keys())
        self.input_dict = data_dict # contains RLE-encoded masks -> to be converted to np.ndarray

        self.target_classes = {
            'large_bowel': 0,
            'small_bowel': 1,
            'stomach': 2
        }
        self.n_classes = 3
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        print(f'Loaded dataset with {len(self.inputs)} samples')


    def __acquire_mask_from_rle(self, img, imname):
        """
        self.target_dict[key_sample] = 
         {'large_bowel': '30043 13 30400 17 30757 21 ...',
          'small_bowel': '35136 7 35495 11 35854 13...',
          'stomach':     '34752 9 35110 12 35469 14 ...'}
        """
        h, w = img.shape[:2]

        mask = np.zeros((h, w))
        for k, rle_str in self.input_dict[imname]['segm'].items():
            val = self.target_classes[k]

            rle_arr = np.array([int(x) for x in rle_str.split()]).reshape(-1, 2)
            for rle in rle_arr:
                x, l = rle
                i, j = x // w, x % w
                if j + l >= w:
                    mask[i, j:w] = val
                    mask[i+1, :j + l - w + 1] = val
                else:
                    mask[i, j:j+l] = val
        return mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        imname = self.inputs[index]

        # Load input and target
        x = cv2.imread(self.input_dict[imname]['path'], -1).astype(np.float32)[..., np.newaxis]
        x /= x.max()
        y = self.__acquire_mask_from_rle(x, imname)
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return x, y