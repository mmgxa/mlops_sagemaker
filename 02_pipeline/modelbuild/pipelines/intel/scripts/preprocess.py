import os
import argparse
import sys
import subprocess
import glob
import shutil
import boto3

from collections import Counter
from git.repo.base import Repo
from smexperiments.tracker import Tracker
from utils import extract_archive
from sklearn.model_selection import train_test_split
from pathlib import Path


git_user = os.environ.get('GIT_USER', "-")
git_email = os.environ.get('GIT_EMAIL', "-")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "intel.zip"
git_path = ml_root / "sagemaker-intel"


s3r = boto3.resource('s3')
bucket = s3r.Bucket('sagemaker-us-west-2-***')
prefix = 'annotated/'

def configure_git():
    subprocess.check_call(['git', 'config', '--global', 'user.email', f'"{git_email}"'])
    subprocess.check_call(['git', 'config', '--global', 'user.name', f'"{git_user}"'])
    


def write_dataset(image_paths, output_dir):
    for img_path in image_paths:
        Path(output_dir / img_path.parent.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, output_dir / img_path.parent.stem / img_path.name)

def down_annotations(dataset_extracted):
    for object in bucket.objects.filter(Prefix = 'annotated'):
        object_key = object.key
        if '.' in object.key:
            bucket.download_file(object.key, f"{dataset_extracted.absolute()}/intel/{object_key.split('/')[1]}/{object_key.split('/')[2]}")
        
def generate_train_test_split():
    dataset_extracted = ml_root / "tmp"
    dataset_extracted.mkdir(parents=True, exist_ok=True)
    
    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(
        from_path=dataset_zip,
        to_path=dataset_extracted
    )
    
    dataset_full = list((dataset_extracted / "intel").glob("*/*.jpg"))
    labels = [x.parent.stem for x in dataset_full]
    
    print(":: Dataset Class Counts: (no annotations)", Counter(labels))
    
    down_annotations(dataset_extracted)
    
    dataset_full = list((dataset_extracted / "intel").glob("*/*.jpg"))
    labels = [x.parent.stem for x in dataset_full]
    
    print(":: Dataset Class Counts: (with annotations)", Counter(labels))
    
    d_train, d_test = train_test_split(dataset_full, stratify=labels)
    
    print("\t:: Train Dataset Class Counts: ", Counter(x.parent.stem for x in d_train))
    print("\t:: Test Dataset Class Counts: ", Counter(x.parent.stem for x in d_test))
    
    for path in ['train', 'test']:
        output_dir = git_path / "dataset" / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Writing Datasets")
    write_dataset(d_train, git_path / "dataset" / "train")
    write_dataset(d_test, git_path / "dataset" / "test")
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # setup git
    print(":: Configuring Git")
    configure_git()
    
    print(":: Generate Train Test Split")
    # extract the input zip file and split into train and test
    generate_train_test_split()
        
    print(":: copy data to train")
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-intel/dataset/train/* /opt/ml/processing/dataset/train', shell=True)
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-intel/dataset/test/* /opt/ml/processing/dataset/test', shell=True)
    
