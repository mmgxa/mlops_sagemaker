from pathlib import Path
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet34
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_integrated_gradients(
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    explain_folder
) -> None:
    """To explain the model using IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        image_tensor, target=pred_label_idx, n_steps=100
    )

    a = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
        use_pyplot=False,
        title='IntegratedGradients'
    )
    a[0].savefig(explain_folder /'a_integ_grad.png') 


def get_noise_tunnel(
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    explain_folder
) -> None:
    """To explain the model using Noise tunnel with IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        image_tensor, nt_samples=1, nt_type="smoothgrad", target=pred_label_idx
    )

    b = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False
    )
    b[0].savefig(explain_folder /'b_integ_grad_noise.png')


def get_shap(
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    explain_folder
) -> None:
    """To explain the model using SHAP."""

    gradient_shap = GradientShap(model)

    rand_img_dist = torch.cat([image_tensor * 0, image_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        image_tensor, n_samples=50, stdevs=0.0001, baselines=rand_img_dist, target=pred_label_idx
    )

    c = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False
    )
    c[0].savefig(explain_folder /'c_grad_shap.png')

def get_occlusion(
    model,
    image_tensor: torch.Tensor,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    explain_folder
) -> None:
    """To explain the model using Occlusion."""

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        image_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    d = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False
    )
    d[0].savefig(explain_folder /'d_occlusion.png')

def get_saliency(model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, explain_folder) -> None:
    """To explain the model using Saliency."""

    saliency = Saliency(model)
    grads = saliency.attribute(image_tensor, target=pred_label_idx)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

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

    original_image = np.transpose(
        inv_transform(image_tensor).squeeze(0).cpu().detach().numpy(), (1, 2, 0)
    )

    _ = viz.visualize_image_attr(
        None, original_image, method="original_image", title="Original Image"
    )
    e = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlaid Gradient Magnitudes",
        use_pyplot=False
    )
    e[0].savefig(explain_folder /'e_saliency.png')

def get_gradcam(model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, explain_folder) -> None:
    """To explain the model using GradCAM."""

    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)  # ,use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

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

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(explain_folder /'f_gradcam.png', visualization)
    


def get_gradcamplusplus(model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, explain_folder) -> None:
    """To explain the model using GradCAM++"""

    target_layers = [model.model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  # , use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

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

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(explain_folder /'g_gradcampp.png', visualization)


def explain_model(model, explain_folder) -> None:
    
    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=mean, std=std)
    
    urllib.request.urlretrieve("https://github.com/mmgxa/E2_7/raw/main/153.jpg", "153.jpg")
    image = Image.open("153.jpg")
    transformed_img = transforms(image)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)


    output = model(image_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )


    get_integrated_gradients(
        model, image_tensor, default_cmap, transformed_img, pred_label_idx, explain_folder
    )

    get_noise_tunnel(model, image_tensor, default_cmap, transformed_img, pred_label_idx, explain_folder)

    get_shap(model, image_tensor, default_cmap, transformed_img, pred_label_idx, explain_folder)

    get_occlusion(model, image_tensor, transformed_img, pred_label_idx, explain_folder)

    image_tensor_grad = image_tensor
    image_tensor_grad.requires_grad = True

    get_saliency(model, image_tensor_grad, pred_label_idx, explain_folder)
    get_gradcam(model, image_tensor_grad, pred_label_idx, explain_folder)
    get_gradcamplusplus(model, image_tensor_grad, pred_label_idx, explain_folder)
        

    file_path = explain_folder / "explanation.md"
    with open(file_path, "w") as f:
        f.write("# Model Explanation  \n")
        f.write("## Integrated Gradients  \n")
        f.write("![](./a_integ_grad.png)\n\n")
        f.write("## Noise Tunnel  \n")
        f.write("![](./b_integ_grad_noise.png)\n\n")
        f.write("## SHAP  \n")
        f.write("![](./c_grad_shap.png)\n\n")
        f.write("## Occlusion  \n")
        f.write("![](./d_occlusion.png)\n\n")
        f.write("## Saliency  \n")
        f.write("![](./e_saliency.png)\n\n")
        f.write("## GradCAM  \n")
        f.write("![](./f_gradcam.png)\n\n")
        f.write("## GradCAM++  \n")
        f.write("![](./g_gradcampp.png)\n\n")