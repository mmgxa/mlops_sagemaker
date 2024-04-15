import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import loggers

import optuna

from model import LitResnet
from dataset import IntelDataModule

from utils import extract_archive
from sklearn.model_selection import train_test_split
from pathlib import Path

DEVICE = "gpu"
EPOCHS = 1
num_cpus = os.cpu_count()


def write_dataset(image_paths, output_dir):
    for img_path in image_paths:
        Path(output_dir / img_path.parent.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, output_dir / img_path.parent.stem / img_path.name)
        

dataset_zip = Path("./intel.zip")
dataset_extracted=Path("./data")

print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
extract_archive(
        from_path=dataset_zip,
        to_path=dataset_extracted
    )
    
dataset_full = list((dataset_extracted / "intel/").glob("*/*.jpg"))
labels = [x.parent.stem for x in dataset_full]

d_train, d_test = train_test_split(dataset_full, stratify=labels)

for path in ['train', 'test']:
    output_dir = Path(".") / path
    print(f"\t:: Creating Directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
print(":: Writing Datasets")
write_dataset(d_train, Path(".") / "train")
write_dataset(d_test, Path(".") / "test")


def run_training(params):
    
    datamodule = IntelDataModule(train_data_dir='./train/', test_data_dir='./test/', num_workers=num_cpus)
    datamodule.setup()
    
    module = LitResnet(params['model_name'], params['lr'], params['optimizer_name'], num_classes=datamodule.num_classes)
    tb_logger = loggers.TensorBoardLogger(save_dir='./tensorboard/')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        logger=[tb_logger],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=False
    )
    
    trainer.fit(module, datamodule)
    

    return trainer.callback_metrics["valid_acc_epoch"].item()

def objective(trial):
    params = {
                "optimizer_name": trial.suggest_categorical('optimizer_name',["SGD", "Adam"]),
                "model_name": trial.suggest_categorical('model_name',["resnet18", "resnet34", "resnet26"]),
                "lr": trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            }
    return run_training(params)

if __name__ == "__main__":
    # run_training(fold = 0)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

