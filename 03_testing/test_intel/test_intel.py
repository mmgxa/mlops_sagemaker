import pytest


import boto3
import sagemaker

import torch
import torchvision.transforms as T

import numpy as np

from PIL import Image


boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)

from sagemaker.pytorch import PyTorchPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = PyTorchPredictor(
    endpoint_name="emlo-capstone-prod",
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transforms = T.Compose(
    [
            T.ToTensor(),
            T.Normalize(mean, std),
    ]
)

classnames = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

@pytest.mark.parametrize("img", classnames)
def test_cifar10_http(img):
    
    inp_img1 = Image.open(f'./test_intel/img/{img}.jpg')
    inp_img1.resize((224, 224))
    inp_img1 = np.array(inp_img1)
    inp_img1 = transforms(inp_img1)
    input_array1 = {"inputs": inp_img1[None, ...].numpy().tolist()}
    out = predictor.predict(input_array1)
    out_t = torch.tensor(out)
    print(f"{img}.jpg is predicted as {classnames[torch.argmax(out_t, dim=-1)[0]]}")
    assert classnames[torch.argmax(out_t, dim=-1)[0]] == img
