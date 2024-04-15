from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

import albumentations as A  
from albumentations.pytorch import ToTensorV2

import numpy as np


class IntelDataset(Dataset):

    def __init__(self, dataset, root, transform=None):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(self.root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        image = self.transform(image=np.array(image))
        label = torch.from_numpy(np.array(label))
        image = (image['image'])
        return (image, label)
    
    def find_classes(self, directory):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
class IntelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str = "data/",
        test_data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        
        self.train_transforms = A.Compose([A.Resize(224,224),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.GaussianBlur(p=0.2),
                                A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                ToTensorV2()])

        self.val_transforms = A.Compose([A.Resize(224,224),
                                    A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                    ToTensorV2()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.classes)
    
    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset_unaug = ImageFolder(self.train_data_dir)
            valset_unaug = ImageFolder(self.test_data_dir)
            testset_unaug = ImageFolder(self.test_data_dir)
            
            trainset = IntelDataset(dataset=trainset_unaug, root=self.train_data_dir, transform=self.train_transforms)
            valset = IntelDataset(dataset=valset_unaug, root=self.test_data_dir, transform=self.val_transforms)
            testset = IntelDataset(dataset=testset_unaug, root=self.test_data_dir, transform=self.val_transforms)
            
            self.data_train, self.data_val, self.data_test = trainset, valset, testset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

