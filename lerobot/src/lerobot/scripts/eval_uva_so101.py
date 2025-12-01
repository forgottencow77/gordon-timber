#!/usr/bin/env python
#
# Evaluate a UVA checkpoint on a LeRobot v3 dataset (SO-101 default).
# Computes average training loss over the validation split.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from lerobot.uva_bridge.dataset import UVALeRobotDataset, _default_action_indices, _default_image_indices


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "lerobot").exists():
            return parent
    return Path.cwd()


def _add_uva_to_path(repo_root: Path) -> Path:
    uva_root = repo_root.parent / "unified_video_action"
    if uva_root.exists():
        sys.path.append(str(uva_root))
        return uva_root
    raise FileNotFoundError(f"Expected UVA repo at {uva_root}")


def build_shape_meta(image_shape, action_dim, img_len, act_len):
    return OmegaConf.create(
        {
            "image_resolution": image_shape[1],
            "obs": {
                "image": {
                    "shape": list(image_shape),
                    "horizon": img_len,
                    "latency_steps": 0.0,
                    "down_sample_steps": 1,
                    "type": "rgb",
                    "ignore_by_policy": False,
                }
            },
            "action": {
                "shape": [action_dim],
                "horizon": act_len,
                "latency_steps": 0.0,
                "down_sample_steps": 1,
                "rotation_rep": "none",
            },
        }
    )


def format_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    obs = {k: v.to(device=device, dtype=torch.float32) for k, v in batch["obs"].items()}
    action = batch["action"].to(device=device, dtype=torch.float32)
    return {"obs": obs, "action": action}


def main():
    parser = argparse.ArgumentParser(description="Evaluate UVA checkpoint on a LeRobot dataset.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image_key", default="observation.images.front")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--image_indices", nargs="+", type=int, default=None)
    parser.add_argument("--action_indices", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    uva_root = _add_uva_to_path(repo_root)
    from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy
    policy_defaults = OmegaConf.load(uva_root / "unified_video_action" / "config" / "model" / "uva.yaml").policy

    image_indices = args.image_indices if args.image_indices is not None else _default_image_indices()
    action_indices = args.action_indices if args.action_indices is not None else _default_action_indices()

    val_ds = UVALeRobotDataset(
        args.repo_id,
        root=args.root,
        image_key=args.image_key,
        action_key=args.action_key,
        image_delta_indices=image_indices,
        action_delta_indices=action_indices,
        split="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    shape_meta = build_shape_meta(
        image_shape=val_ds.image_shape,
        action_dim=val_ds.action_dim,
        img_len=len(image_indices),
        act_len=len(action_indices),
    )

    policy_kwargs = {
        "vae_model_params": policy_defaults.vae_model_params,
        "autoregressive_model_params": policy_defaults.autoregressive_model_params,
        "action_model_params": {**policy_defaults.action_model_params, "predict_action": True},
        "shape_meta": shape_meta,
        "n_action_steps": policy_defaults.n_action_steps,
        "shift_action": True,
        "language_emb_model": None,
        "task_name": "umi_so101",
        "task_modes": ["policy_model"],
        "normalizer_type": "none",
        "use_history_action": False,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "different_history_freq": False,
        "action_mask_ratio": policy_defaults.action_mask_ratio,
    }

    device = torch.device(args.device)
    policy = UnifiedVideoActionPolicy(**policy_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["model_state"], strict=False)
    policy.eval()

    losses: list[float] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="eval"):
            batch = format_batch(batch, device=device)
            loss, _ = policy(batch)
            losses.append(loss.item())

    avg = sum(losses) / max(1, len(losses))
    print(f"Average loss on split 'val': {avg:.4f} over {len(losses)} batches")


if __name__ == "__main__":
    main()
