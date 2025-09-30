import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_dataset import BaseDataset
from ..dataset_util import read_depth, read_image_cv2_alpha


_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class ObjaverseDataset(BaseDataset):
    """Dataset adapter that wraps Objaverse orthographic assets for VGGT."""

    def __init__(
        self,
        common_conf,
        root: str,
        split: str = "train",
        min_num_images: int = 2,
        len_train: int = 100_000,
        len_test: int = 10_000,
        assume_opengl_coords: bool = True,
        max_sequences: Optional[int] = None,
        train_val_split: Optional[float] = None,
        split_seed: int = 17,
    ) -> None:
        super().__init__(common_conf=common_conf)

        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Objaverse root does not exist: {self.root}")

        self.split = split
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.assume_opengl_coords = assume_opengl_coords
        self.train_val_split = train_val_split
        self.split_seed = split_seed

        if self.train_val_split is not None:
            if not (0.0 < self.train_val_split < 1.0):
                raise ValueError(
                    "train_val_split must be in the open interval (0, 1) to avoid leaks"
                )

        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self._cfg_len_train = len_train
        self._cfg_len_test = len_test

        meta_files = sorted(self.root.rglob("meta.json"))
        if not meta_files:
            raise FileNotFoundError(f"No meta.json found under {self.root}")

        if max_sequences is not None:
            meta_files = meta_files[:max_sequences]
        if self.debug:
            meta_files = meta_files[: min(4, len(meta_files))]

        self.sequence_list: List[str] = []
        self.data_store: Dict[str, List[Dict[str, Any]]] = {}
        self.sequence_dirs: Dict[str, Path] = {}

        total_frames = 0
        for meta_path in meta_files:
            seq_dir = meta_path.parent
            seq_name = str(seq_dir.relative_to(self.root))
            if not self._sequence_in_split(seq_name):
                continue

            frames = self._parse_meta(meta_path)
            if len(frames) < min_num_images:
                continue

            self.sequence_list.append(seq_name)
            self.sequence_dirs[seq_name] = seq_dir
            self.data_store[seq_name] = frames
            total_frames += len(frames)

        if not self.sequence_list:
            raise RuntimeError(
                f"No valid Objaverse sequences found in {self.root} with min_num_images={min_num_images}"
            )

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frames

        if split == "train":
            configured_len = self._cfg_len_train
        else:
            configured_len = self._cfg_len_test

        if configured_len is None or configured_len <= 0:
            # Default to one epoch covering every available sequence once
            self.len_train = self.sequence_list_len
        else:
            self.len_train = configured_len

        logging.info(
            "%s: Objaverse sequences=%d total_frames=%d",
            "Training" if self.training else "Testing",
            self.sequence_list_len,
            self.total_frame_num,
        )

    def _sequence_hash_value(self, seq_name: str) -> float:
        key = f"{self.split_seed}:{seq_name}".encode("utf-8")
        digest = hashlib.sha1(key).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return (value % 10**12) / 10**12

    def _sequence_in_split(self, seq_name: str) -> bool:
        if self.train_val_split is None:
            return True

        val = self._sequence_hash_value(seq_name)
        if self.split == "train":
            return val < self.train_val_split
        else:
            return val >= self.train_val_split

    def _parse_meta(self, meta_path: Path) -> List[Dict[str, Any]]:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        frames: List[Dict[str, Any]] = []
        for loc in meta.get("locations", []):
            frame_index = int(loc.get("index", len(frames)))
            color_name = None
            depth_name = None
            for frame in loc.get("frames", []):
                ftype = frame.get("type", "").lower()
                if ftype == "color":
                    color_name = frame.get("name")
                elif ftype == "depth":
                    depth_name = frame.get("name")

            if color_name is None:
                continue

            color_path = meta_path.parent / color_name
            depth_path = meta_path.parent / depth_name if depth_name else None
            if not color_path.exists():
                logging.warning("Missing color frame: %s", color_path)
                continue
            if self.load_depth and (depth_path is None or not depth_path.exists()):
                logging.warning("Missing depth frame for %s", color_path)
                depth_path = None

            transform_matrix = np.asarray(loc.get("transform_matrix"), dtype=np.float32)
            if transform_matrix.shape != (4, 4):
                logging.warning("Invalid transform matrix shape for %s", color_path)
                continue

            extrinsic = self._transform_to_extrinsic(transform_matrix)

            frames.append(
                {
                    "index": frame_index,
                    "color_path": color_path,
                    "depth_path": depth_path,
                    "extrinsic": extrinsic,
                    "ortho_scale": float(loc.get("ortho_scale", 1.0)),
                }
            )

        frames.sort(key=lambda x: x["index"])
        return frames

    def _transform_to_extrinsic(self, cam_to_world: np.ndarray) -> np.ndarray:
        cam_to_world_mat = cam_to_world.astype(np.float32, copy=False)
        if self.assume_opengl_coords:
            cam_to_world_mat = _GL_TO_CV @ cam_to_world_mat @ _GL_TO_CV
        world_to_cam = np.linalg.inv(cam_to_world_mat)
        return world_to_cam[:3, :].astype(np.float32)

    def _build_intrinsic(self, image_shape: np.ndarray, ortho_scale: float) -> np.ndarray:
        height, width = image_shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image shape for intrinsic computation")

        pixel_size_x = ortho_scale / float(width)
        # camera height in world units follows Blender convention
        pixel_size_y = ortho_scale * (height / float(width)) / float(height)

        fx = 1.0 / max(pixel_size_x, 1e-8)
        fy = 1.0 / max(pixel_size_y, 1e-8)
        cx = float(width) * 0.5
        cy = float(height) * 0.5

        intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return intrinsic

    def get_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[List[int]] = None,
        aspect_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            if seq_index is None:
                raise ValueError("Either seq_index or seq_name must be provided")
            seq_name = self.sequence_list[seq_index]

        frames = self.data_store[seq_name]
        available = len(frames)

        # Always consume the leading views (Objaverse renders are ordered around the object).
        desired_views = min(4, available)
        if desired_views == 0:
            raise RuntimeError(f"Sequence {seq_name} does not contain any frames")

        if img_per_seq is not None:
            desired_views = min(desired_views, int(img_per_seq))

        ids = np.arange(desired_views, dtype=np.int32)

        target_shape = self.get_target_shape(aspect_ratio)

        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        cam_points: List[np.ndarray] = []
        world_points: List[np.ndarray] = []
        point_masks: List[np.ndarray] = []
        image_paths: List[str] = []
        original_sizes: List[np.ndarray] = []
        used_ids: List[int] = []

        for frame_id in ids:
            frame = frames[int(frame_id)]
            color_path = frame["color_path"]
            depth_path = frame.get("depth_path")

            image = read_image_cv2_alpha(str(color_path), background=(0, 0, 0))
            if image is None:
                logging.warning("Failed to read image %s", color_path)
                continue

            depth_map = None
            if self.load_depth and depth_path is not None:
                depth_map = read_depth(str(depth_path))
                depth_map[depth_map >= depth_map.max()] = 0.0

            original_size = np.array(image.shape[:2])
            intrinsic = self._build_intrinsic(image.shape, frame["ortho_scale"])
            extrinsic = frame["extrinsic"]

            (
                image,
                depth_map,
                extrinsic,
                intrinsic,
                _,
                _,
                _,
                _,
            ) = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extri_opencv=extrinsic,
                intri_opencv=intrinsic,
                original_size=original_size,
                target_image_shape=target_shape,
                filepath=str(color_path),
            )

            if depth_map is None:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                cam_pts = np.zeros((*image.shape[:2], 3), dtype=np.float32)
                world_pts = np.zeros_like(cam_pts)
                point_mask = np.zeros(image.shape[:2], dtype=bool)
            else:
                depth_map = depth_map.astype(np.float32)
                cam_pts, point_mask = self._depth_to_cam_points_ortho(depth_map, intrinsic)
                world_pts = self._cam_to_world(cam_pts, extrinsic)
                cam_pts[~point_mask] = 0.0
                world_pts[~point_mask] = 0.0

            images.append(np.ascontiguousarray(image))
            depths.append(np.ascontiguousarray(depth_map))
            extrinsics.append(extrinsic.astype(np.float32))
            intrinsics.append(intrinsic.astype(np.float32))
            cam_points.append(np.ascontiguousarray(cam_pts))
            world_points.append(np.ascontiguousarray(world_pts))
            point_masks.append(np.ascontiguousarray(point_mask))
            image_paths.append(str(color_path))
            original_sizes.append(original_size.astype(np.int32))
            used_ids.append(int(frame_id))

        if not images:
            raise RuntimeError(f"No frames could be loaded for sequence {seq_name}")

        batch = {
            "seq_name": f"objaverse_{seq_name}",
            "ids": np.array(used_ids, dtype=np.int32),
            "frame_num": len(used_ids),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        return batch

    def _depth_to_cam_points_ortho(
        self, depth_map: np.ndarray, intrinsic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth_map.shape
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        pixel_size_x = 1.0 / max(fx, 1e-8)
        pixel_size_y = 1.0 / max(fy, 1e-8)

        u, v = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        x_cam = (u - cx) * pixel_size_x
        y_cam = (v - cy) * pixel_size_y
        z_cam = depth_map

        cam_points = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
        point_mask = (z_cam > 0.0).astype(bool)
        return cam_points, point_mask

    def _cam_to_world(self, cam_points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        R = extrinsic[:, :3]
        t = extrinsic[:, 3]
        cam_flat = cam_points.reshape(-1, 3)
        world_flat = (cam_flat - t) @ R.T
        return world_flat.reshape(cam_points.shape).astype(np.float32)
