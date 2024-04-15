import json
import numpy as np

import torch
import torchvision.transforms as T

from utils import decode_base64_to_image

import boto3
import sagemaker

boto_session = boto3.Session(region_name='us-west-2')
sagemaker_session = sagemaker.Session(boto_session=boto_session)


from sagemaker.pytorch import PyTorchPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transforms = T.Compose(
    [
            T.ToTensor(),
            T.Normalize(mean, std),
    ]
)


response_headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

classnames = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def lambda_handler(event, context):
    img_b64 = event["body"]
    
    predictor = PyTorchPredictor(
        endpoint_name="emlo-capstone-prod",
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    img_b64 = event["body"]

    try:
        inp_img1 = decode_base64_to_image(img_b64)
        
        inp_img1.resize((224, 224))
        inp_img1 = np.array(inp_img1)
        inp_img1 = transforms(inp_img1)
        input_array1 = {"inputs": inp_img1[None, ...].numpy().tolist()}
        out = predictor.predict(input_array1)
        out_t = torch.tensor(out)
        print(out_t.numpy())


        label = classnames[torch.argmax(out_t, dim=-1)[0]]
        print(label)

        return {
            "statusCode": 200,
            "headers": response_headers,
            "body": label,
        }

    except Exception as e:
        print(e)

        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"message": "Failed to process image: {}".format(e)}),
        }
