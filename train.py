import argparse
import cv2
import json
import os
import pandas as pd
import time
import numpy as np
from glob import glob
from tqdm import tqdm

import pdb
import sys

def excepthook(type, value, traceback):
    pdb.post_mortem(traceback)

# excepthook.old = sys.excepthook
# sys.excepthook = excepthook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
import cfg
import utils

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=cfg.NUM_EPOCHS, help='Number of epochs to train')
    parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=cfg.LR, help='Learning rate')
    parser.add_argument('-md', '--model-depth', type=int, required=False, default=cfg.INIT_MODEL_DEPTH, help='Initial depth of UNet model')
    parser.add_argument('-bs', '--batch-size', type=int, required=False, default=cfg.BATCH_SIZE, help='Batch size')
    parser.add_argument('-d', '--data-path', type=str, required=False, default=cfg.TRAIN_DATA_PATH, help='Path to data (either for train or for test')
    parser.add_argument('--validate', action='store_true', required=False, default=False, help='Validation mode')
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=False, help='Path to the trained model, either to continue trainig or for validation')
    parser.add_argument('--output',, type=str, required=False, help='Path to the generated csv file')
    return parser.parse_args()

from utils import dice
from model import UNet
from data import load_train_data, normalize

class Trainer:
    def __init__(self, model, device, model_name) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.loss = dice
        # self.outfunc = torch.nn.Sigmoid()
        self.outfunc = torch.nn.Softmax(dim=1)
        self.model_name = model_name
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name, exist_ok=True)
    
    def save_checkpoint(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        torch.save(state_dict, open(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o775), 'wb'))

    def train(self, train_gen, val_gen, epochs=128, lr=0.01):
        # TODO Add logger
        # add optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        for e in range(1, epochs+1):
            start_time = time.time()
            # train
            train_losses=[]
            for i, batch in enumerate(train_gen):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                # forward
                logits = self.model(x)
                preds = self.outfunc(logits)

                # compute loss
                loss = self.loss(preds, y)
                loss.backward()
                train_losses.append(loss.detach().cpu().numpy())
                if i %  5 == 0:
                    print(f'[epoch {e:03d}] Step {i:03d} | dice loss: {loss.detach().cpu().numpy():.3f} (avg 50 batch: {np.mean(train_losses):.3f})')
                    train_losses = []
                # optimizer.step
                self.optim.step()
                self.optim.zero_grad()
                if i == 0:
                    utils.save_mask_img(x[0], preds[0], f'{self.model_name}/epoch_{e:03d}.png')
                    utils.save_img(x[0], f'{self.model_name}/epoch_{e:03d}_img.png', mode='img')
                    utils.save_img(preds[0], f'{self.model_name}/epoch_{e:03d}_mask.png', mode='mask')
                    utils.save_img(
                        (one_hot(y[0].long(), num_classes=4) * 255).detach().cpu().numpy().astype('uint8')[..., 1:],
                        f'{self.model_name}/epoch_{e:03d}_gtmask.png',
                        mode='mask'
                    )
                    print(preds[0].sum())
                    print(y[0].sum())

            train_time = time.time()
            print(f'[epoch {e:03d}] Epoch took {train_time-start_time}s')

            # val
            with torch.no_grad():
                val_losses = []
                for i, batch in enumerate(val_gen):
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = self.model(x)
                    preds = self.outfunc(logits)
                    loss = self.loss(preds, y)
                    val_losses.append(loss.cpu().detach().numpy())
                print(f'[epoch {e:03d}] Val finished | dice loss: {np.mean(val_losses):.3f}')
            all_time = time.time()
            print(f'[epoch {e:03d}] Val took {all_time-train_time}s')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.save_checkpoint(f'{self.model_name}/epoch_{e:03d}_dice_{np.mean(val_losses):.3f}.pth')


def train(args):
    train_df = prepare_train_dataframe(args.data_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dl, test_dl = load_train_data(train_df, batch_size=args.batch_size, resize_size=cfg.RESIZE_SIZE)
    model = UNet(n_classes=4, model_depth=args.model_depth)

    model_name=f'exp_lr_{args.learning_rate}_unet_size_{cfg.RESIZE_SIZE}_depth_{args.model_depth}'

    trainer = Trainer(model=model, device=device, model_name=model_name)
    trainer.train(train_dl, test_dl, epochs=args.epochs, lr=args.learning_rate)


def load_model(ckpt_path, model_depth, n_classes):
    model = UNet(n_classes=n_classes, model_depth=model_depth)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    return model


def validate(args):
    sm =  torch.nn.Softmax(dim=0)
    n_classes = 4

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = load_model(args.checkpoint, args.model_depth, n_classes).to(device)

    data_path = cfg.VAL_DATA_PATH
    fs = glob(f'{data_path}/**/*.png', recursive=True)
    print(f'loading {len(fs)} imgs')

    ids = []
    classes = []
    segs = []

    for f in tqdm(fs):
        if '_mask' in f: continue
        imid = os.path.basename(f)[:10]
        imid = os.path.basename(os.path.dirname(os.path.dirname(f))) + '_' + imid
        # read img
        im = cv2.imread(f, -1).astype('float')
        h, w = im.shape[:2]
        im, _ = normalize(im, None)
        # split by parts
        crop_size = 224
        nh, nw = (h + crop_size - 1) // crop_size, (w + crop_size - 1) // crop_size

        batch_list = []
        slicing = []
        for i in range(nh):
            for j in range(nw):
                hslice = slice(crop_size * i, crop_size * (i + 1))
                wslice = slice(crop_size * j, crop_size * (j + 1))
                if i == nh - 1: # last height crop
                    hslice = slice(h - crop_size, h)
                if j == nw - 1: # last height crop
                    wslice = slice(w - crop_size, w)
                batch_list.append(im[hslice, wslice])
                slicing.append([hslice, wslice])

        # eval each part
        with torch.no_grad():
            batch = torch.tensor(np.array(batch_list)).unsqueeze(1).to(dtype=torch.float32, device=device)
            preds = model(batch)

        # glue back together, overlaying parts -> mean()
        res = torch.zeros((n_classes, h, w))
        counts = torch.zeros((h, w))
        for pred, slices in zip(preds, slicing):
            hslice, wslice = slices
            counts[hslice, wslice] += 1
            res[:, hslice, wslice] += pred
        res /= counts.unsqueeze(0)
        np_res = sm(res).round().long().argmax(dim=0).detach().cpu().numpy()


        # some visualisations for debug
        # utils.save_mask_img(
        #     torch.tensor(im[np.newaxis, ...]),
        #     sm(res),
        #     os.path.join(f'{data_path}', 'preds', f'{imid}_mask.png'))

        # some postproc
        # TODO

        # calc mask
        mask_res = np_res[1:, ...]
        for cl, val in cfg.TARGET_CLASSES.items():
            rle_str = utils.mask_to_rle_str(np_res, val)
            ids.append(imid)
            classes.append(cl)
            segs.append(rle_str)

    res_df_dict ={ 
        'id' : ids,
        'class' : classes,
        'predicted' : segs,
    }
    df = pd.DataFrame.from_dict(res_df_dict)
    df.to_csv(args.output), index=False)


def prepare_train_dataframe(data_path):
    df = pd.read_csv(f'{data_path}/train.csv')
    df = df[df.id.str.startswith('case101')]
    notnull_df = df[df.segmentation.notnull()].reset_index()

    fs = glob(f'{data_path}/case101/**/*.png', recursive=True)
    # fs = glob(f'{data_path}/train/**/*.png', recursive=True)
    print(len(fs))
    id_to_fn_mapping = {os.path.basename(os.path.dirname(os.path.dirname(f))) + '_' + os.path.basename(f)[:10]: f for f in fs}

    df_mapped = {}
    grouped = notnull_df.groupby(['id'])
    for x in notnull_df.id.value_counts().index:
        df_mapped[x] = {'path': id_to_fn_mapping[x], 'segm': {}}
        records = grouped.get_group(x)
        for _, r in records.iterrows():
            df_mapped[x]['segm'][r['class']] = r['segmentation']
    return df_mapped


if __name__ == '__main__':
    args = parse_args()

    if args.validate:
        validate(args)
    else:
        train(args)
