#!/usr/bin/env python
#
# Lightweight adapter to feed LeRobot v3 datasets into the UVA training
# pipeline. It wraps `lerobot.datasets.lerobot_dataset.LeRobotDataset`
# and reshapes samples to match UVA's expected structure:
#   {
#       "obs": {"image": FloatTensor[T_img, C, H, W], "img_indices": FloatTensor[T_img, 1]},
#       "action": FloatTensor[T_act, Da]
#   }
# No normalization is applied here; UVA's policy handles it internally.
#
# The adapter relies on `delta_timestamps` to fetch fixed windows of
# past/future frames, mimicking UVA's original indexing:
#   - Images: 8 frames at indices [-12, -8, -4, 0, 4, 8, 12, 16]
#   - Actions: 32 frames at indices [-15, ..., 16]
#
# The class is intentionally minimal and agnostic to the underlying
# robot; the caller chooses which camera/state keys to use.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# UVA's default image size expected by the MAR model
UVA_DEFAULT_IMAGE_SIZE = 256


def _default_image_indices() -> list[int]:
    # Same stride/length as UMI's `umi_lazy_dataset`: 8 frames over ~1.5s window.
    return list(range(-12, 17, 4))


def _default_action_indices() -> list[int]:
    # UVA's DiffActLoss expects exactly 16 action timesteps.
    # With shift_action=True and use_history_action=False, the trajectory becomes
    # nactions[:, T//2-1:-1] where T=8 (image frames), i.e. nactions[:, 3:-1].
    # To get 16 actions after slicing: len(nactions) - 3 - 1 = 16 → len(nactions) = 20
    return list(range(-3, 17))  # 20 action frames: indices -3 to 16 inclusive


@dataclass
class UVALeRobotDataset(Dataset):
    """Return UVA‑style batches from a LeRobot v3 dataset.

    Parameters
    ----------
    repo_id: str
        Hugging Face dataset id or local path understood by LeRobotDataset.
    root: str | Path | None
        Optional local cache root for the dataset.
    image_key: str
        Feature name of the camera to use (e.g. 'observation.images.front').
    action_key: str
        Feature name containing the action vector (usually 'action').
    image_delta_indices: Sequence[int]
        Relative frame offsets (in dataset steps) to fetch for images.
    action_delta_indices: Sequence[int]
        Relative frame offsets (in dataset steps) to fetch for actions.
    split_ratio: float
        Fraction of frames used for training when `split="train"`.
    split: str
        'train' or 'val'. Split is performed by slicing the dataset length.
    target_image_size: int
        Target image size (square) for resizing. UVA expects 256x256 by default.
    """

    repo_id: str
    root: str | Path | None = None
    image_key: str = "observation.images.front"
    action_key: str = "action"
    image_delta_indices: Sequence[int] = None  # type: ignore[assignment]
    action_delta_indices: Sequence[int] = None  # type: ignore[assignment]
    split_ratio: float = 0.95
    split: str = "train"
    target_image_size: int = UVA_DEFAULT_IMAGE_SIZE

    def __post_init__(self):
        meta = LeRobotDatasetMetadata(self.repo_id, root=self.root)
        fps = meta.fps

        # Validate provided feature keys early to give actionable errors.
        if self.image_key not in meta.features:
            image_keys = [k for k in meta.features if "image" in k]
            raise ValueError(
                f"image_key '{self.image_key}' not found in dataset features. "
                f"Available image-like keys: {image_keys}"
            )
        if self.action_key not in meta.features:
            action_keys = [k for k in meta.features if "action" in k]
            raise ValueError(
                f"action_key '{self.action_key}' not found in dataset features. "
                f"Available action-like keys: {action_keys}"
            )

        img_idx = list(_default_image_indices() if self.image_delta_indices is None else self.image_delta_indices)
        act_idx = list(_default_action_indices() if self.action_delta_indices is None else self.action_delta_indices)

        # Convert index offsets to seconds for LeRobotDataset delta_timestamps API.
        delta_timestamps = {
            self.image_key: [i / fps for i in img_idx],
            self.action_key: [i / fps for i in act_idx],
        }

        self.dataset = LeRobotDataset(
            self.repo_id,
            root=self.root,
            delta_timestamps=delta_timestamps,
            revision=meta.revision,
        )

        # Determine frame slice for the requested split.
        total = len(self.dataset)
        train_len = int(total * self.split_ratio)
        if self.split == "train":
            self._index_range = (0, train_len)
        elif self.split == "val":
            self._index_range = (train_len, total)
        else:
            raise ValueError(f"Unsupported split '{self.split}', expected 'train' or 'val'.")

        self.image_delta_indices = img_idx
        self.action_delta_indices = act_idx
        self.meta = meta

    def __len__(self) -> int:
        start, end = self._index_range
        return max(0, end - start)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start, end = self._index_range
        global_idx = start + idx
        if global_idx >= end:
            raise IndexError(f"Index {idx} out of bounds for split '{self.split}' with length {len(self)}")

        sample = self.dataset[global_idx]

        images = sample[self.image_key]  # (T_img, C, H, W)
        actions = sample[self.action_key]  # (T_act, Da)
        if actions.ndim == 1:  # fallback when delta_timestamps not provided for action
            actions = actions.unsqueeze(0)

        # Resize images to target size if needed (UVA expects 256x256 by default)
        images = images.float()
        _, _, h, w = images.shape
        if h != self.target_image_size or w != self.target_image_size:
            images = F.interpolate(
                images,
                size=(self.target_image_size, self.target_image_size),
                mode="bilinear",
                align_corners=False,
            )

        obs = {"image": images}
        # Optional: provide the sampled frame indices so UVA can skip its own subsampling.
        obs["img_indices"] = torch.tensor(self.image_delta_indices, dtype=torch.float32).view(-1, 1)
        return {
            "obs": obs,
            "action": actions.float(),
            "task": sample.get("task", ""),
            "timestamp": sample.get("timestamp", torch.tensor(0.0)),
        }

    @property
    def action_dim(self) -> int:
        act_shape = self.meta.features[self.action_key]["shape"]
        return act_shape[-1] if isinstance(act_shape, Iterable) else int(act_shape)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return (C, H, W) after resizing to target_image_size."""
        feat = self.meta.features[self.image_key]
        shape = feat["shape"]  # [H, W, C] or [C, H, W] depending on dataset
        # LeRobot stores as [H, W, C] in metadata but returns [C, H, W] tensors
        # We return the shape after resize: (C, target_size, target_size)
        c = shape[2] if len(shape) == 3 and shape[2] <= 4 else shape[0]
        return (c, self.target_image_size, self.target_image_size)
