import torch

from dataset import IntelDataModule

import albumentations as A

import json
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import IntelDataModule

from functools import partial

from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift

torch.manual_seed(42)
torch.cuda.manual_seed(42)

import os
num_cpus = os.cpu_count()

ml_root = Path("/opt/ml")
dataset_dir = ml_root / "processing" / "test"

def data_drift_func(model, dd_folder):
    
    datamodule = IntelDataModule(
        train_data_dir=dataset_dir.absolute(),
        test_data_dir=dataset_dir.absolute(),
        num_workers=0,
        batch_size = 100,
    )
    datamodule.setup()

    test_dl = next(iter(datamodule.test_dataloader()))
    preprocess_fn = partial(preprocess_drift, model=model, device=device, batch_size=100)
    cd = MMDDrift(test_dl[0], backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn, n_permutations=100)

    dd_unpert = cd.predict(test_dl[0][:100])
    dd_unpert['data']['distance_threshold'] = dd_unpert['data']['distance_threshold'].item()


    perturb = A.RandomBrightnessContrast(
        brightness_limit=1.5,
        contrast_limit=0.9,
        p=1.0
    )

    perturbed_images = []

    for idx in range(100):
        perturbed_image = torch.tensor(
            perturb(
                image=test_dl[0][idx].numpy(),
            )['image']
        )

        perturbed_images.append(perturbed_image)

    perturbed_images = torch.stack(perturbed_images)

    dd_pert = cd.predict(perturbed_images[:100])
    dd_pert['data']['distance_threshold'] = dd_pert['data']['distance_threshold'].item()


    outfile_unpert = dd_folder / "dd_unpert.json"
    with outfile_unpert.open("w") as f:
        f.write(json.dumps(dd_unpert))
        
        
    outfile_pert = dd_folder / "dd_pert.json"
    with outfile_pert.open("w") as f:
        f.write(json.dumps(dd_pert))