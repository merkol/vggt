# dataset/objaverse_dataloader.py
from __future__ import annotations

import json
import os
from pathlib import Path
from re import I
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from transformations import build_transforms  # returns builders we use below
from utils import (
    PointCloudProcessor,
    extrinsic_to_posevec_scipy,
    posevec_to_extrinsic,
    unnormalize_rgb,
    custom_collate_fn,
)

__all__ = ["ObjaverseOrthoDataset"]


def rgba_tensor_to_rgb_and_mask(
    rgba_t: torch.Tensor, background: Tuple[float, float, float] = (1.0, 1.0, 1.0)
):
    """
    Keeps your exact logic:
      rgba_t: (4, H, W) in [0,1]
    Returns:
      rgb:  (3, H, W) in [0,1] with alpha blending over 'background'
      mask: (1, H, W) binary from alpha > 0.5
    """
    if rgba_t.shape[0] != 4:
        raise ValueError("Expected tensor shape (4, H, W) for RGBA image")

    rgb = rgba_t[:3]  # (3,H,W)
    alpha_soft = rgba_t[3:4]  # (1,H,W)

    bg = torch.tensor(background, device=rgba_t.device, dtype=rgba_t.dtype).view(
        3, 1, 1
    )
    out_rgb = (rgb * alpha_soft + bg * (1 - alpha_soft)).clamp(0, 1)

    alpha_mask = (alpha_soft > 0.5).float()
    return out_rgb, alpha_mask


class ObjaverseOrthoDataset(Dataset):
    """
    Returns:
      images:          (V,3,H,W)  [0,1] optionally augmented; feed to VGGT (VGGT normalizes internally)
      depth:           (V,1,H,W)  float32 metric
      mask:            (V,1,H,W)  {0,1}
      camera_context:  (V,7)      [w,x,y,z, tx,ty,tz] per view
      scales:          (V,)
      pcd:             (N,3) or (N,6) if colors present
      object_id:       str
    """

    def __init__(
        self,
        root: str | Path,
        num_views: int = 4,
        view_ids: Optional[Sequence[int]] = (0, 1, 2, 3),
        ref_idx: int = 0,
        image_size: int = 224,
        # Optional: pass custom transforms; otherwise defaults are built
        to_rgba_unit_tensor: Optional[
            T.Compose
        ] = None,  # Resize + ToTensor for RGBA → (4,H,W) in [0,1]
        color_aug: Optional[T.Compose] = None,  # Photometric augs on [0,1] 3ch
        normalize: Optional[T.Normalize] = None,  # Ignored; VGGT normalizes internally
        depth_transform: Optional[
            T.Compose
        ] = None,  # NEAREST + to tensor float [1,H,W]
        cache_meta: bool = True,
        max_instances: Optional[int] = None,
        percentage: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_views = int(num_views)
        self.view_ids = None if view_ids is None else list(view_ids)
        self.ref_idx = int(ref_idx)
        self.image_size = int(image_size)
        self.cache_meta = cache_meta

        # Build defaults from transformations helper
        (
            dft_to_rgba_unit_tensor,  # Resize(bilinear) + ToTensor() -> (4,H,W) [0,1]
            dft_color_aug,  # photometric augs for [0,1] RGB tensors
            dft_normalize,  # ImageNet norm
            dft_depth_transform,
        ) = build_transforms(self.image_size)

        self.to_rgba_unit_tensor = to_rgba_unit_tensor or dft_to_rgba_unit_tensor
        self.color_aug = color_aug if color_aug is not None else dft_color_aug
        # VGGT Aggregator handles normalization internally; keep here for compatibility but do not apply
        self.normalize = None
        self.depth_transform = depth_transform or dft_depth_transform

        # Gather objects (folders with meta.json)
        self._obj_dirs = [m.parent for m in self.root.rglob("meta.json")]
        if not self._obj_dirs:
            raise FileNotFoundError(f"No meta.json found under {self.root}")

        total = len(self._obj_dirs)
        if percentage is not None:
            if not (0.0 <= percentage <= 1.0):
                raise ValueError("percentage must be in [0,1]")
            max_instances = int(total * percentage)

        if max_instances is not None:
            max_instances = max(1, min(int(max_instances), total))
            self._obj_dirs = self._obj_dirs[:max_instances]

        self._meta_cache: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._obj_dirs)

    def _load_meta(self, obj_dir: Path) -> Dict[str, Any]:
        if not self.cache_meta:
            with open(obj_dir / "meta.json", "r") as f:
                return json.load(f)
        key = str(obj_dir)
        if key not in self._meta_cache:
            with open(obj_dir / "meta.json", "r") as f:
                self._meta_cache[key] = json.load(f)
        return self._meta_cache[key]

    @staticmethod
    def _remove_background_depth(depth_img: Image.Image) -> Image.Image:
        depth_arr = np.asarray(depth_img).copy()
        mx = depth_arr.max()
        depth_arr[depth_arr >= mx] = 0
        return Image.fromarray(depth_arr)

    @staticmethod
    def _read_rgba_pil(path: Path) -> Image.Image:
        return Image.open(path).convert("RGBA")

    @staticmethod
    def _read_depth_pil(path: Path) -> Image.Image:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        depth_img = Image.fromarray(arr.astype(np.float32))
        return ObjaverseOrthoDataset._remove_background_depth(depth_img)

    def __getitem__(self, index: int):
        obj_dir = self._obj_dirs[index]
        meta = self._load_meta(obj_dir)

        # Camera extrinsics (10,4,4)
        trans = np.asarray([loc["transform_matrix"] for loc in meta["locations"]])

        # View selection
        if self.view_ids is None:
            ids = np.random.choice(10, self.num_views, replace=False)
            view_ids = list(sorted(ids))
        else:
            view_ids = self.view_ids[: self.num_views]

        pose_vecs = extrinsic_to_posevec_scipy(trans[view_ids])  # (V,7) [w,x,y,z, tx,ty,tz]

        images_list, depth_list, mask_list, scales = (
            [],
            [],
            [],
            [],
            [],
        )

        for vid in view_ids:
            # --- RGB(A) ---
            rgba_pil = self._read_rgba_pil(obj_dir / f"color_{vid:04d}.webp")
            # Resize + ToTensor() -> (4,H,W) in [0,1]
            rgba_unit = self.to_rgba_unit_tensor(rgba_pil)  # [0,1], shape (4,H,W)

            # Split
            rgb_unit, mask_bin = rgba_tensor_to_rgb_and_mask(rgba_unit)  # both tensors

            # Optional: ensure mask is exactly 0/1 (already thresholded)
            mask_t = mask_bin

            # Color augment on [0,1] 3-channel tensor
            if self.color_aug is not None:
                rgb_aug = self.color_aug(rgb_unit)
            else:
                rgb_aug = rgb_unit

            # Do NOT normalize here; VGGT will normalize internally

            # --- Depth ---
            depth_pil = self._read_depth_pil(obj_dir / f"depth_{vid:04d}.exr")
            depth_t = self.depth_transform(depth_pil)  # (1,H,W) float32 metric

            # --- Scale ---
            scales.append(meta["locations"][vid]["ortho_scale"])

            # Stash
            images_list.append(rgb_aug)
            depth_list.append(depth_t)
            mask_list.append(mask_t)

        # Stack to (V, ..)
        images = torch.stack(images_list, dim=0)  # (V,3,H,W) in [0,1], augmented if color_aug is set
        depth = torch.stack(depth_list, dim=0)  # (V,1,H,W)
        mask = torch.stack(mask_list, dim=0)  # (V,1,H,W)
        pose_vecs = torch.as_tensor(pose_vecs, dtype=torch.float32)  # (V,7)
        scales = torch.as_tensor(scales, dtype=torch.float32)  # (V,)

        # --- Ground-truth PCD (optional colors) ---
        pcd_path = obj_dir / "point_cloud.ply"
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_np = np.asarray(pcd.points)  # (N,3)
        if pcd.has_colors():
            pcd_cols = np.asarray(pcd.colors)  # (N,3) [0,1]
            pcd_np = np.concatenate([pcd_np, pcd_cols], axis=1)  # (N,6)
        pcd_tensor = torch.from_numpy(pcd_np).float()

        return {
            "images": images,
            "depths": depth,           # align with training/loss expected key
            "mask": mask,
            "point_masks": mask,       # reuse segmentation as valid pixel mask
            "camera_context": pose_vecs,
            "scales": scales,
            "pcd": pcd_tensor,
            "object_id": obj_dir.name,
        }


# -------------------------- quick sanity run -------------------------- #
if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    torch.manual_seed(23)
    np.random.seed(23)
    from tqdm import tqdm

    image_size = 224
    # Build defaults
    to_rgba_unit_tensor, color_aug, normalize, depth_tf = build_transforms(image_size)
    ds = ObjaverseOrthoDataset(
        root="/home/vgl/objaverse-nn/data/mnt/pfs/data/texture_ortho10view_easylight_objaverse",
        num_views=4,
        view_ids=[0, 1, 2, 3],
        image_size=image_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    for _, batch in tqdm(enumerate(dl)):
        # print("object_id:", batch["object_id"])
        # print(
        #     "rgb_model:",
        #     batch["rgb_model"].shape,
        #     batch["rgb_model"].min().item(),
        #     batch["rgb_model"].max().item(),
        # )
        # print(
        #     "rgb_clean:",
        #     batch["rgb_clean"].shape,
        #     batch["rgb_clean"].min().item(),
        #     batch["rgb_clean"].max().item(),
        # )
        # print(
        #     "depth:",
        #     batch["depth"].shape,
        #     batch["depth"].min().item(),
        #     batch["depth"].max().item(),
        # )
        # print("mask:", batch["mask"].shape, torch.unique(batch["mask"]))
        # print(
        #     "pose_vecs:",
        #     batch["pose_vecs"].shape,
        #     batch["pose_vecs"].min().item(),
        #     batch["pose_vecs"].max().item(),
        # )

        # from utils.utils import unnormalize_rgb
        # ## Save depth, mask and rgb from first batch and first view
        # rgb = unnormalize_rgb(batch["rgb_clean"][0])
        # rgb = (rgb[0] * 255).byte().permute(1, 2, 0).cpu().numpy()
        # Image.fromarray(rgb).save("rgb_clean.png")

        # depth = batch["depth"][0][0]
        # depth = (depth / depth.max() * 255).byte().cpu().numpy()
        # Image.fromarray(depth[0]).save("depth.png")

        # mask = (batch["mask"][0][0] * 255).byte().cpu().numpy()
        # Image.fromarray(mask[0]).save("mask.png")
        # create point cloud from depth and rgb_clean

        transform_matrices = posevec_to_extrinsic(batch["camera_context"])
        processor = PointCloudProcessor()
        for i in range(len(batch["object_id"])):
            points, colors = processor.process_views(
                depth_images=batch["depths"][i].to(device),
                rgb_images=batch["images"][i].movedim(1, -1).to(device),
                transform_matrices=transform_matrices[i].to(device),
                ortho_scales=batch["scales"][i].to(device),
                masks=batch["mask"][i].to(device),
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            path = Path(
                "/home/vgl/objaverse-nn/data/mnt/pfs/data/texture_ortho10view_easylight_objaverse"
            )
            path_f = path / batch["object_id"][i][:2]
            o3d.io.write_point_cloud(
                path_f / batch["object_id"][i] / "point_cloud.ply", pcd
            )
