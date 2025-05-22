import os
import timm, torch
import torch.nn as nn
from datasets import VLASS
from torch.utils.data import random_split 
import lightning as L

def running_on_windows():
    return os.name == 'nt'

class VLASSLoader(L.LightningDataModule):
    def __init__(
        self, root: str, batch_size: int = 64,
        num_workers: int = 5, pin_memory: bool = True,
        transform = None, test_ratio: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transform
        # windows does not support multiprocessing
        if running_on_windows():
            self.num_workers = 0
        else:
            self.num_workers = num_workers

    def setup(self, stage: str):
        # dataset = VLASS(
        #     root = self.hparams.root,
        #     transform = self.transform
        # )
        
        # test_size = int(self.hparams.test_ratio * len(dataset))
        # train_size = len(dataset) - test_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

        self.train_dataset = VLASS(
            root = self.hparams.root, split='train', 
            transform = self.transform
        )
        self.val_dataset = VLASS(
            root = self.hparams.root, split='test', 
            transform = self.transform
        )
        
    def train_dataloader(self):
        print(f'Train size: {len(self.train_dataset)}')
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
        
    def val_dataloader(self):
        print(f'Val size: {len(self.val_dataset)}')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )