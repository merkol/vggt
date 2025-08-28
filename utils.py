import torch
import torch.nn.functional as F

from kaolin.metrics.pointcloud import chamfer_distance
from torch.utils.data.dataloader import default_collate
from typing import Optional, Tuple
import torch.distributed as dist
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


class PointCloudProcessor:
    def __init__(
        self,
        voxel_size: float = 0.01,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        device: str = "cuda:0",
        mask_threshold: float = 0.5,
    ):
        """Initialize the point cloud processor with processing parameters.

        Args:
            voxel_size: Voxel size for downsampling
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio for outlier removal
            device: PyTorch device to use (e.g., "cuda:0", "cpu")
            mask_threshold: Threshold for mask values to consider as valid points
        """
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.mask_threshold = mask_threshold

        # Set up PyTorch device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def process_depth_view(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        transform_matrix: torch.Tensor,
        ortho_scale: float,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single depth view and return colored point cloud using GPU tensors.

        Args:
            depth: Depth image tensor (H, W)
            rgb: RGB image tensor (H, W, 3) in range [0, 1]
            transform_matrix: 4x4 transformation matrix tensor
            ortho_scale: Orthographic scale factor
            mask: Optional mask tensor (H, W) or (1, H, W) in range [0, 1]

        Returns:
            Tuple of (points, colors) tensors
        """
        # Ensure tensors are on the correct device
        depth = depth.to(self.device)
        rgb = rgb.to(self.device)
        transform_matrix = transform_matrix.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        # Ensure depth is 2D
        if depth.dim() == 3:
            depth = depth[0, :, :]

        # Ensure mask is 2D if provided
        if mask is not None and mask.dim() == 3:
            mask = mask[0, :, :]

        # Calculate orthographic parameters
        H, W = depth.shape
        aspect_ratio = W / H
        ortho_width = ortho_scale
        ortho_height = ortho_width / aspect_ratio

        # Generate pixel grid using torch
        x = torch.linspace(-ortho_width / 2, ortho_width / 2, W, device=self.device)
        y = torch.linspace(ortho_height / 2, -ortho_height / 2, H, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        # Create points
        depth_corrected = -depth
        points = torch.stack((xx, yy, depth_corrected), dim=-1).reshape(-1, 3)

        # Transform points
        ones = torch.ones(points.shape[0], 1, device=self.device)
        points_homo = torch.cat([points, ones], dim=1)
        points_world = torch.matmul(transform_matrix, points_homo.T).T[:, :3]

        # Add mask-based filtering if mask is provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid_mask = mask_flat > self.mask_threshold
        else:
            valid_mask = torch.ones(
                points_world.shape[0], dtype=torch.bool, device=self.device
            )

        # Filter valid points and their colors
        valid_points = points_world[valid_mask]
        valid_colors = rgb.reshape(-1, 3)[valid_mask]

        return valid_points, valid_colors

    def process_views(
        self,
        depth_images: torch.Tensor,
        rgb_images: torch.Tensor,
        transform_matrices: torch.Tensor,
        ortho_scales: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process multiple views and combine them into a single point cloud using GPU.

        Args:
            depth_images: Array of depth images (N, H, W) or (N, 1, H, W)
            rgb_images: Array of RGB images (N, H, W, 3)
            transform_matrices: Array of transformation matrices (N, 4, 4)
            ortho_scales: Array of orthographic scales (N,)
            masks: Optional array of masks (N, H, W) or (N, 1, H, W)

        Returns:
            Tuple of (points, colors) tensors for the combined point cloud
        """
        all_points = []
        all_colors = []

        # Prepare masks iterator
        if masks is not None:
            mask_iter = masks
        else:
            mask_iter = [None] * len(depth_images)

        # Process each view
        for i, (depth, rgb, transform, scale) in enumerate(
            zip(depth_images, rgb_images, transform_matrices, ortho_scales)
        ):
            mask = mask_iter[i] if masks is not None else None
            points, colors = self.process_depth_view(
                depth, rgb, transform, scale.item(), mask
            )

            # Only add points if we have valid points
            if points.shape[0] > 0:
                all_points.append(points)
                all_colors.append(colors)

        # Check if we have any valid points
        if not all_points:
            # Return empty tensors if no valid points found
            empty_points = torch.empty((0, 3), device=self.device)
            empty_colors = torch.empty((0, 3), device=self.device)
            return empty_points, empty_colors

        # Combine all points and colors
        combined_points = torch.cat(all_points, dim=0)
        combined_colors = torch.cat(all_colors, dim=0)

        # Apply voxel downsampling open3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(combined_colors.detach().cpu().numpy())
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_points = torch.from_numpy(np.asarray(pcd.points)).to(self.device)
        combined_colors = torch.from_numpy(np.asarray(pcd.colors)).to(self.device)

        return combined_points, combined_colors


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with variable-sized point clouds.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary with batched data, where point clouds are kept as a list
        due to variable sizes
    """
    # Extract point clouds separately (they have variable sizes)
    pcds = [item["pcd"] for item in batch]

    # Remove 'pcd' from each item temporarily for default collation
    batch_without_pcd = []
    for item in batch:
        item_copy = {k: v for k, v in item.items() if k != "pcd"}
        batch_without_pcd.append(item_copy)

    # Use default collate for everything except point clouds
    collated_batch = default_collate(batch_without_pcd)

    # Add point clouds as a list (not stacked due to variable sizes)
    collated_batch["pcd"] = pcds

    return collated_batch


def unnormalize_rgb(
    rgb: torch.Tensor,
    mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]),
    std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]),
) -> torch.Tensor:
    """
    Unnormalize RGB images.

    Args:
        rgb: Normalized RGB images (B, 3, H, W)
        mean: Mean values for normalization
        std: Standard deviation values for normalization

    Returns:
        Unnormalized RGB images (B, 3, H, W)
    """
    return torch.clamp(
        rgb * std.view(1, 3, 1, 1).to(rgb.device)
        + mean.view(1, 3, 1, 1).to(rgb.device),
        0,
        1,
    )


def extrinsic_to_posevec_scipy(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert extrinsic(s) E = [[R|t],[0 0 0 1]] (camera-to-world) to pose vec(s)
    as (w,x,y,z, tx,ty,tz).

    Args:
        extrinsic: (4,4) or (V,4,4) array (c2w). If you have w2c, invert first.

    Returns:
        pose_vec: (7,) for (4,4) input, or (V,7) for (V,4,4) input
    """
    extrinsic = np.asarray(extrinsic)
    assert extrinsic.shape[-2:] == (4, 4), "Input must be (...,4,4)"

    # Support single or batch
    single = extrinsic.ndim == 2
    if single:
        extrinsic = extrinsic[None, ...]  # (1,4,4)

    rot = extrinsic[..., :3, :3]  # (V,3,3)
    trans = extrinsic[..., :3, 3]  # (V,3)

    quat_xyzw = R.from_matrix(rot).as_quat()  # (V,4) in (x,y,z,w)
    quat_wxyz = np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)

    # Normalize & canonicalize qw >= 0
    qnorm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    quat_wxyz = quat_wxyz / np.clip(qnorm, 1e-12, None)
    sign = np.where(quat_wxyz[..., 0:1] < 0, -1.0, 1.0)
    quat_wxyz = quat_wxyz * sign

    pose = np.concatenate([quat_wxyz, trans], axis=-1)  # (V,7)
    return pose[0] if single else pose


@torch.jit.script
def posevec_to_extrinsic(pose_vecs: torch.Tensor) -> torch.Tensor:
    # (B,V,7) -> (B,V,4,4), [w,x,y,z, tx,ty,tz]
    assert pose_vecs.ndim == 3 and pose_vecs.shape[-1] == 7
    device, dtype = pose_vecs.device, pose_vecs.dtype

    q = pose_vecs[..., :4]
    t = pose_vecs[..., 4:7]

    # normalize & canonicalize
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    qw, qx, qy, qz = q.unbind(-1)
    sign = torch.where(qw < 0, -1.0, 1.0)
    qw, qx, qy, qz = qw * sign, qx * sign, qy * sign, qz * sign

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    ).to(dtype)

    B, V = pose_vecs.shape[:2]
    E = torch.eye(4, device=device, dtype=dtype).expand(B, V, 4, 4).clone()
    E[..., :3, :3] = R
    E[..., :3, 3] = t
    return E


def edge_aware_smoothing_loss(
    depth_pred: torch.Tensor,
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_grad: float = 1.0,
) -> torch.Tensor:
    """
    Compute edge-aware smoothing loss for depth prediction.

    This loss encourages smooth depth predictions while preserving edges
    by reducing smoothing penalty near image edges where depth discontinuities
    are expected.

    Args:
        depth_pred: Predicted depth maps (B, V, 1, H, W) or (B, 1, H, W)
        image: RGB images (B, V, 3, H, W) or (B, 3, H, W) in range [0, 1]
        mask: Optional mask (B, V, 1, H, W) or (B, 1, H, W) to ignore certain regions
        lambda_grad: Weight for the gradient penalty

    Returns:
        torch.Tensor: Edge-aware smoothing loss
    """
    # Handle both multi-view (B, V, C, H, W) and single view (B, C, H, W) inputs
    original_shape = depth_pred.shape
    if len(original_shape) == 5:  # Multi-view case (B, V, 1, H, W)
        B, V, _, H, W = depth_pred.shape
        depth_pred = depth_pred.reshape(B * V, 1, H, W)
        image = image.reshape(B * V, 3, H, W)
        if mask is not None:
            mask = mask.reshape(B * V, 1, H, W)

    # Convert RGB to grayscale for edge detection
    # RGB to grayscale weights: 0.299*R + 0.587*G + 0.114*B
    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(
        1, 3, 1, 1
    )
    gray_image = torch.sum(image * gray_weights, dim=1, keepdim=True)  # (B*V, 1, H, W)

    # Compute image gradients using Sobel operators
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device
    ).view(1, 1, 3, 3)

    # Compute image gradients
    img_grad_x = F.conv2d(gray_image, sobel_x, padding=1)
    img_grad_y = F.conv2d(gray_image, sobel_y, padding=1)
    img_grad_magnitude = torch.sqrt(img_grad_x**2 + img_grad_y**2 + 1e-8)

    # Compute depth gradients
    depth_grad_x = F.conv2d(depth_pred, sobel_x, padding=1)
    depth_grad_y = F.conv2d(depth_pred, sobel_y, padding=1)

    # Edge-aware weights: reduce smoothing penalty near image edges
    # Use exponential weighting: exp(-lambda_grad * |∇I|)
    edge_weights = torch.exp(-lambda_grad * img_grad_magnitude)

    # Compute weighted smoothing loss
    smooth_loss_x = edge_weights * torch.abs(depth_grad_x)
    smooth_loss_y = edge_weights * torch.abs(depth_grad_y)

    # Apply mask if provided
    if mask is not None:
        # Create gradient masks by applying conv2d to the mask
        mask_grad_x = F.conv2d(
            mask.float(), torch.ones(1, 1, 3, 1, device=mask.device), padding=(1, 0)
        )
        mask_grad_y = F.conv2d(
            mask.float(), torch.ones(1, 1, 1, 3, device=mask.device), padding=(0, 1)
        )

        # Only consider gradients where both neighboring pixels are valid
        mask_x = mask_grad_x > 1.5  # Both pixels in x-direction are valid
        mask_y = mask_grad_y > 1.5  # Both pixels in y-direction are valid

        smooth_loss_x = smooth_loss_x * mask_x.float()
        smooth_loss_y = smooth_loss_y * mask_y.float()

        # Normalize by valid gradient count
        total_loss = (smooth_loss_x.sum() + smooth_loss_y.sum()) / (
            mask_x.sum() + mask_y.sum() + 1e-8
        )
    else:
        # Average over all pixels
        total_loss = smooth_loss_x.mean() + smooth_loss_y.mean()

    return total_loss


def chamfer_distance_variable_size(pred_pcds, gt_pcds, device: torch.device):
    """
    Compute chamfer distance for variable-sized point clouds.

    Args:
        pred_pcds: List of predicted point cloud tensors (each is a tuple of (points, colors))
        gt_pcds: List of ground truth point cloud tensors
        device: Device to move tensors to

    Returns:
        chamfer_loss: Average chamfer distance across the batch
    """

    batch_losses = []
    for pred_pcd, gt_pcd in zip(pred_pcds, gt_pcds):
        # Extract points from the tuple (points, colors) - we only need points for chamfer distance
        if isinstance(pred_pcd, tuple):
            pred_points = pred_pcd[0]  # Get points tensor
        else:
            pred_points = pred_pcd

        # Ensure tensors are on the correct device and have batch dimension
        pred_points = pred_points.to(dtype=torch.float32, device=device).unsqueeze(
            0
        )  # (1, N, 3)
        gt_pcd = gt_pcd.unsqueeze(0).to(dtype=torch.float32, device=device)[
            :, :, :3
        ]  # (1, M, 3)

        # Compute chamfer distance for this pair
        loss = chamfer_distance(pred_points, gt_pcd)
        batch_losses.append(loss)

    # Average across the batch
    return torch.stack(batch_losses).mean()


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    experiment_name,
    config,
    wandb_run_id=None,
    checkpoint_dir=None,
):
    """Save training checkpoint"""
    # Access the underlying model from DDP wrapper (handle compiled model)
    if hasattr(model, "_orig_mod"):  # For torch.compile
        if hasattr(model._orig_mod, "module"):  # For DDP
            model_state = model._orig_mod.module.state_dict()
        else:
            model_state = model._orig_mod.state_dict()
    elif hasattr(model, "module"):  # For DDP without compile
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
        "wandb_run_id": wandb_run_id,
        "experiment_name": experiment_name,
    }

    # Create checkpoint filename
    checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"

    # Use provided directory or current directory
    if checkpoint_dir:
        import os

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    else:
        checkpoint_path = checkpoint_filename

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state - handle DDP and compiled models
    if hasattr(model, "_orig_mod"):  # For torch.compile
        if hasattr(model._orig_mod, "module"):  # For DDP
            model._orig_mod.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
    elif hasattr(model, "module"):  # For DDP without compile
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint.get("wandb_run_id"),
        checkpoint.get("experiment_name", "default"),
    )


def setup_ddp():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_rank():
    """Get the rank of the current process."""
    return int(os.environ.get("RANK", 0))


def get_world_size():
    """Get the total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def prefer_flash_attention():
    """
    Prefer FlashAttention if available (Ampere/Ada + fp16/bf16 + constraints).
    Otherwise let PyTorch fall back to mem-efficient or math SDPA.
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)  # enable Flash backend
        torch.backends.cuda.enable_mem_efficient_sdp(
            True
        )  # keep mem-efficient as fallback
        torch.backends.cuda.enable_math_sdp(True)  # and math as last fallback
    except Exception:
        pass


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous in memory to avoid DDP stride warnings."""
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(
        embed_dim // 2, pos_flat[:, 0], omega_0=omega_0
    )  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(
        embed_dim // 2, pos_flat[:, 1], omega_0=omega_0
    )  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(
        embed_dim // 2,
        dtype=torch.float32 if device.type == "mps" else torch.double,
        device=device,
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


if __name__ == "__main__":
    pass
