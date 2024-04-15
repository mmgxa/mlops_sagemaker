from pathlib import Path
import urllib
from typing import List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet34

from albumentations.pytorch import ToTensorV2
from captum.attr import FeatureAblation
from captum.robust import FGSM, PGD, MinParamPerturbation
from PIL import Image

ml_root = Path("/opt/ml")
robust_folder = ml_root / "processing" / "robustness"
robust_folder.mkdir(parents=True, exist_ok=True)

def get_prediction(model, image: torch.Tensor) -> Tuple[str, float, int]:
    """Function to return the model prediction, confidence and label index."""

    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    with torch.no_grad():
        output = model(image)

    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item(), pred_label_idx.item()


def image_save(img: torch.Tensor, pred: str, title) -> None:
    """Function to display the image with prediction."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()

    plt.imsave(f'{robust_folder}/{title}.png', npimg)


def get_pgd(model, image_tensor: torch.Tensor, target_index: int) -> None:
    """Function to create a targeted PGD adversarial image."""

    pgd = PGD(
        model, torch.nn.CrossEntropyLoss(reduction="none"), lower_bound=-1, upper_bound=1
    )  # construct the PGD attacker
    perturbed_image_pgd = pgd.perturb(
        inputs=image_tensor,
        radius=0.13,
        step_size=0.02,
        step_num=7,
        target=torch.tensor([target_index]),
        targeted=True,
    )

    new_pred_pgd, score_pgd, _ = get_prediction(model, perturbed_image_pgd)

    image_save(perturbed_image_pgd, new_pred_pgd + " " + str(score_pgd), "a_PGD")


def get_fgsm(model, image_tensor: torch.Tensor, target_index: int) -> None:
    """Function to create a non-targeted FGSM adversarial image."""

    # Construct FGSM attacker
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(image_tensor, epsilon=0.16, target=target_index)

    new_pred_fgsm, score_fgsm, _ = get_prediction(model, perturbed_image_fgsm)

    image_save(perturbed_image_fgsm, new_pred_fgsm + " " + str(score_fgsm), "b_FGSM")


def get_random_noise(model, image: Image):
    """Function to check model robustness by adding random Gaussian noise."""

    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.GaussNoise(p=0.35),
            ToTensorV2(),
        ]
    )
    image = np.array(image)
    image_tensor = transforms(image=image)["image"]
    image_tensor = image_tensor.unsqueeze(0)

    pred, score, index = get_prediction(model, image_tensor)
    print(f"Predicted: {pred} (confidence = {score}, index = {index})")
    image_save(image_tensor, pred + " " + str(score), "c_random_noise")


def get_random_brightness(model, image: Image):
    """Function to check model robustness by adding random brightness and contrast."""

    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    )
    image = np.array(image)
    image_tensor = transforms(image=image)["image"]
    image_tensor = image_tensor.unsqueeze(0)

    pred, score, index = get_prediction(model, image_tensor)
    print(f"Predicted: {pred} (confidence = {score}, index = {index})")
    image_save(image_tensor, pred + " " + str(score), "d_random_brightness")


def model_robustness(model) -> None:
    """Function to check the robustness of any pre-trained timm model
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        None
    """

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    urllib.request.urlretrieve("https://github.com/mmgxa/E2_7/raw/main/153.jpg", "153.jpg")
    image = Image.open("153.jpg")
    transformed_img = transforms(image)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)

    predicted_label, prediction_score, pred_label_idx = get_prediction(model, image_tensor)

    print(
        f"Predicted: {predicted_label} (confidence = {prediction_score}, index = {pred_label_idx})"
    )

    image_tensor_grad = image_tensor
    image_tensor_grad.requires_grad = True
    get_pgd(model, image_tensor_grad, pred_label_idx)

    get_fgsm(model, image_tensor_grad, pred_label_idx)

    get_random_noise(model, image)

    get_random_brightness(model, image)



    file_path = robust_folder / "robustness.md"
    with open(file_path, "w") as f:
        f.write("# Model Robustness  \n")
        f.write("## Projected Gradient Descent  \n")
        f.write("![](./a_PGD.png)\n\n")
        f.write("## Fast Gradient Sign Method  \n")
        f.write("![](./b_FGSM.png)\n\n")
        f.write("## Random Noise  \n")
        f.write("![](./c_random_noise.png)\n\n")
        f.write("## Random Brightness  \n")
        f.write("![](./d_random_brightness.png)\n\n")