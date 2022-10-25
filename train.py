import os
import time
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

torch.manual_seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import dice
from model import UNet
from data import SegmentationDataSet, load_train_data

class Trainer:
    def __init__(self, model, device, title) -> None:
        self.model = model
        self.device = device
        self.loss = dice
        self.title = title
    
    def save_checkpoint(self, path):
        if not os.path.exists(os.dirname(path)):
            os.makedirs(os.dirname(path), exists_ok=True)
        state_dict = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        torch.save(state_dict, open(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o775), 'wb'))

    def train(self, train_gen, val_gen, epochs=128, lr=0.01):
        # add optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        for e in range(1, epochs+1):
            start_time = time.time()
            # train
            for i, batch in enumerate(train_gen):
                x, y = batch
                
                # forward
                logits = self.model(x)
                preds = torch.sigmoid(logits)

                # compute loss
                loss = self.loss(preds, y)
                loss.backward()

                if i % 10 == 0:
                    print(f'Step {i:03d} | dice loss: {loss.detach().cpu().numpy():.3f}')

                # optimizer.step
                self.optim.step()
                self.optim.zero_grad()

            train_time = time.time()
            print(f'Epoch took {train_time-start_time}s')
            # val
            with torch.no_grad():
                val_losses = []
                for i, batch in enumerate(val_gen):
                    x, y = batch
                    logits = self.model(x)
                    preds = torch.sigmoid(logits)
                    loss = self.loss(preds, y)
                    val_losses.append(loss.cpu().detach().numpy())
                print(f'Val finished | dice loss: {np.mean(val_losses):.3f}')
            all_time = time.time()
            print(f'Val took {all_time-train_time}s')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.save_checkpoint(f'{self.title}/epoch_{e:03d}_dice_{np.mean(val_losses):.3f}.pth')

if __name__ == '__main__':
    df = pd.read_csv('train/train.csv')
    df = df[df.id.str.startswith('case101')]
    notnull_df = df[df.segmentation.notnull()].reset_index()
    
    fs = glob('train/**/*.png', recursive=True)
    mapping = {os.path.basename(os.path.dirname(os.path.dirname(f))) + '_' + os.path.basename(f)[:10]: f for f in fs}
    list(mapping.items())[:1]

    df_mapped = {}
    grouped = notnull_df.groupby(['id'])
    for x in notnull_df.id.value_counts().index:
        df_mapped[x] = {'path': mapping[x], 'segm': {}}
        records = grouped.get_group(x)
        for _, r in records.iterrows():
            df_mapped[x]['segm'][r['class']] = r['segmentation']
    
    train_dl, test_dl = load_train_data(df_mapped)
    model = UNet(n_classes=3)
    trainer = Trainer(model=model, device='cpu', title='test0')
    trainer.train(train_dl, test_dl)