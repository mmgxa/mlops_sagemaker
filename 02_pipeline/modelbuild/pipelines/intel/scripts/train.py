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
import smdistributed.dataparallel.torch.torch_smddp
from pytorch_lightning.strategies import DDPStrategy

from model import LitResnet
from dataset import IntelDataModule

os.environ["USE_SMDEBUG"] = "0"

sm_output_dir = 'Path(os.environ.get("SM_OUTPUT_DIR"))'
sm_model_dir = Path(os.environ.get("SM_MODEL_DIR"))
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

train_channel = os.environ.get("SM_CHANNEL_TRAIN")
test_channel = os.environ.get("SM_CHANNEL_TEST")

ml_root = Path("/opt/ml")

model_name = os.environ.get("MODEL_NAME")
batch_size = int(os.environ.get("BATCH_SIZE"))
opt_name = os.environ.get("OPT_NAME")
lr = float(os.environ.get("LR"))
augmentations = os.environ.get("AUGMENTATIONS")

# world_size = int(os.environ["SM_NUM_GPUS"]) * len(os.environ["SM_HOSTS"])
# num_nodes = len(os.environ["SM_HOSTS"])
# num_gpus = int(os.environ["SM_NUM_GPUS"])

world_size = int(os.environ["WORLD_SIZE"])
num_gpus = int(os.environ["SM_NUM_GPUS"])
num_nodes = len(os.environ["SM_HOSTS"])

env = LightningEnvironment()
env.world_size = lambda: int(os.environ.get("WORLD_SIZE", 0))
env.global_rank = lambda: int(os.environ.get("RANK", 0))

ddp = DDPStrategy(
    cluster_environment=env, 
    process_group_backend="smddp", 
    accelerator="gpu"
    )

def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)
    
    return sm_training_env


def train(model, datamodule, sm_training_env):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"])
    
    trainer = pl.Trainer(
        max_epochs=10,
        num_nodes=num_nodes,
        devices=num_gpus,
        logger=[tb_logger],
        num_sanity_val_steps=0,
        strategy=ddp,
        log_every_n_steps=1
    )
    
    trainer.fit(model, datamodule)
    
    return trainer

def save_scripted_model(model, output_dir):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, output_dir / "model.scripted.pt")


def save_last_ckpt(trainer, output_dir):
    trainer.save_checkpoint(output_dir / "last.ckpt")


if __name__ == '__main__':
    
    img_dset = ImageFolder(train_channel)
    
    print(":: Classnames: ", img_dset.classes)
    
    datamodule = IntelDataModule(train_data_dir=train_channel,
                                test_data_dir=test_channel,
                                num_workers=num_cpus,
                                batch_size=batch_size)
    datamodule.setup()
    
    model = LitResnet(model=model_name, lr=lr, opt=opt_name, num_classes=datamodule.num_classes)
    
    sm_training_env = get_training_env()
    
    print(":: Training ...")
    trainer = train(model, datamodule, sm_training_env)
    
    print(":: Saving Model Ckpt")
    save_last_ckpt(trainer, sm_model_dir)
    
    print(":: Saving Scripted Model")
    save_scripted_model(model, sm_model_dir)

