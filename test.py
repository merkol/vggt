from training.data.datasets.objaverse import ObjaverseDataset
from addict import Dict  # Hydra’s configs behave like dot dicts
import cv2
import numpy as np

common = Dict(
    {
        "debug": True,
        "training": False,
        "get_nearby": False,
        "load_depth": True,
        "inside_random": False,
        "allow_duplicate_img": False,
        "img_size": 224,
        "patch_size": 14,
        "rescale": True,
        "rescale_aug": False,
        "landscape_check": False,
    }
)

ds = ObjaverseDataset(
    common_conf=common,
    root="/home/vgl/objaverse-nn/data/mnt/pfs/data/texture_ortho10view_easylight_objaverse",
    split="test",
    min_num_images=4,
    max_sequences=2,  # keep tiny while debugging
)

sample = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
print(sample["images"][0].shape, sample["depths"][0].shape)
## print min and max depth

print(np.min(sample["depths"][0]), np.max(sample["depths"][0]))

## save the first image and depth

# cv2.imwrite("test_image.png", sample["images"][0][:, :, ::-1])
# cv2.imwrite("test_image_1.png", sample["images"][1][:, :, ::-1])

### print sample dict keys
print(sample.keys())
### print cam matrices
print(sample["intrinsics"][0])
print(sample["extrinsics"][0])
print(sample["world_points"][0].shape)
print(sample["point_masks"][0].shape)
print(sample["original_sizes"][0])
print(sample["image_paths"][0])
print(sample["ids"])
print(sample["seq_name"])
print(sample["frame_num"])
print(sample["cam_points"][0].shape)

## save one point mask
# cv2.imwrite("test_mask.png", (sample["point_masks"][0] * 255).astype(np.uint8))
### save one depth map
# depth_map = sample["depths"][0]
# ## save
# depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255.0
# cv2.imwrite("test_depth.png", depth_map.astype(np.uint8))


# from vggt.models.vggt import VGGT

# model = VGGT(
#     img_size=224,
#     patch_size=14,
#     embed_dim=384,
#     enable_point=False,
#     enable_track=False,
#     aggregator_cfg={"depth": 12, "num_heads": 6, "patch_embed": "dinov2_vits14_reg"},
# )
# print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
