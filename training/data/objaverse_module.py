# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Any

from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path so we can import the dataset and utils
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from objaverse_dataloader import ObjaverseOrthoDataset
from utils import custom_collate_fn


class ObjaverseDataModule:
    """
    Minimal Hydra-friendly wrapper that exposes get_loader() for the Trainer.

    Produces batches compatible with the Trainer expectations:
    - images: (B,S,3,H,W) in [0,1]
    - depths: (B,S,1,H,W)
    - point_masks: (B,S,1,H,W)
    - camera_context: (B,S,7)
    - object_id, pcd, scales are passed through for optional use/visualization
    """

    def __init__(
        self,
        *,
        root: str,
        image_size: int = 224,
        num_views: int = 4,
        view_ids: Optional[Sequence[int]] = (0, 1, 2, 3),
        batch_size: int = 8,
        num_workers: int = 8,
        shuffle: bool = True,
        percentage: Optional[float] = None,
        max_instances: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        subset_start: Optional[float] = None,
        subset_end: Optional[float] = None,
        **unused_kwargs: Any,
    ) -> None:
        self.root = root
        self.image_size = int(image_size)
        self.num_views = int(num_views)
        self.view_ids = None if view_ids is None else list(view_ids)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.percentage = percentage
        self.max_instances = max_instances
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.subset_start = subset_start
        self.subset_end = subset_end

        # Gracefully ignore any unexpected keyword arguments introduced by
        # upstream config composition (e.g., inherited from other dataset configs)
        if unused_kwargs:
            try:
                import warnings
                warnings.warn(
                    f"ObjaverseDataModule: ignoring unknown init args: {list(unused_kwargs.keys())}",
                    RuntimeWarning,
                )
            except Exception:
                pass

    def get_loader(self, epoch: int):  # epoch kept for API compatibility
        dataset = ObjaverseOrthoDataset(
            root=self.root,
            num_views=self.num_views,
            view_ids=self.view_ids,
            image_size=self.image_size,
            percentage=self.percentage,
            max_instances=self.max_instances,
        )

        # Optional fractional subset split when train/val share the same root
        if self.subset_start is not None or self.subset_end is not None:
            s = 0.0 if self.subset_start is None else float(self.subset_start)
            e = 1.0 if self.subset_end is None else float(self.subset_end)
            if not (0.0 <= s <= 1.0 and 0.0 <= e <= 1.0 and s < e):
                raise ValueError("subset_start/subset_end must satisfy 0<=start<end<=1")
            total = len(dataset)
            start_idx = int(total * s)
            end_idx = int(total * e)
            indices = list(range(start_idx, max(start_idx + 1, end_idx)))
            dataset = Subset(dataset, indices)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=custom_collate_fn,
        )


