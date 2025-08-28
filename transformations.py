# tensor-friendly augmentations with Kornia
import torch
import numpy as np
from torchvision import transforms as T
import kornia.augmentation as K


class TensorColorAug:
    """Apply Kornia augs to a single CxHxW float tensor in [0,1]."""

    def __init__(self):
        self.augs = K.AugmentationSequential(
            K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8
            ),
            K.RandomAutoContrast(p=0.1),
            K.RandomEqualize(p=0.05),
            K.RandomGrayscale(p=0.05),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            data_keys=["input"],
            same_on_batch=False,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3, H, W) float32 in [0,1]
        return self.augs(x.unsqueeze(0)).squeeze(0)


def build_transforms(img_size: int):
    to_rgba_unit_tensor = T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # keeps 4 channels if input PIL is RGBA → (4,H,W) in [0,1]
        ]
    )

    color_aug = TensorColorAug()  # << use Kornia here

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    depth_transform = T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(
                lambda x: torch.from_numpy(np.asarray(x).copy()).unsqueeze(0).float()
            ),
        ]
    )

    # return the same 4-tuple you use in your dataloader
    return to_rgba_unit_tensor, color_aug, normalize, depth_transform
