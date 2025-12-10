#!/usr/bin/env python
#
# Run UVA policy in inference mode and dump predicted actions (and optionally input frames).
#
# Example:
#   python src/lerobot/scripts/show_uva_predictions.py \\
#       --checkpoint outputs/uva_so101/epoch0010.pt \\
#       --repo_id tenkau/record-test \\
#       --image_key observation.images.laptop \\
#       --max_batches 2 --save_frames --device cuda

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Inspect UVA policy outputs on a few batches.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image_key", default="observation.images.front")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--image_indices", nargs="+", type=int, default=None)
    parser.add_argument("--action_indices", nargs="+", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=1, help="Number of batches to sample for inspection.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_frames", action="store_true", help="Save the input frames used for each sample.")
    parser.add_argument("--out_dir", type=Path, default=Path("uva_outputs"))
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="Use safe weights-only checkpoint loading. "
        "Default loads the full checkpoint (weights_only=False) for compatibility with older UVA saves.",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    uva_root = _add_uva_to_path(repo_root)
    from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

    policy_defaults = OmegaConf.load(uva_root / "unified_video_action" / "config" / "model" / "uva.yaml").policy

    if args.image_indices is not None:
        image_indices = args.image_indices
    else:
        # The UVA config expects 4 conditioning frames (see uva.yaml comment).
        default_img_idx = _default_image_indices()
        image_indices = default_img_idx[:4] if len(default_img_idx) > 4 else default_img_idx
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
    loader = DataLoader(
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
        # Keep behavior aligned with UVA defaults
        "selected_training_mode": policy_defaults.selected_training_mode,
        "use_history_action": False,
        "use_proprioception": False,
        "predict_wrist_img": False,
        "predict_proprioception": False,
        "different_history_freq": False,
        "action_mask_ratio": policy_defaults.action_mask_ratio,
    }

    device = torch.device(args.device)
    policy = UnifiedVideoActionPolicy(**policy_kwargs).to(device).eval()

    if args.weights_only:
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=args.weights_only or False)
    policy.load_state_dict(ckpt["model_state"], strict=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "predictions.jsonl"
    records: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        obs = {k: v.to(device) for k, v in batch["obs"].items()}
        gt_action = batch["action"].to(device)

        with torch.no_grad():
            out = policy.predict_action(obs)

        pred_action = out["action"]  # (B, n_action_steps, Da)
        # Align shapes for metrics: clamp to shared length
        shared_steps = min(gt_action.shape[1], pred_action.shape[1])
        shared_gt = gt_action[:, :shared_steps]
        shared_pred = pred_action[:, :shared_steps]
        mae = torch.mean(torch.abs(shared_gt - shared_pred)).item()
        mse = torch.mean((shared_gt - shared_pred) ** 2).item()

        rec = {
            "batch_index": batch_idx,
            "gt_action_shape": list(gt_action.shape),
            "pred_action_shape": list(pred_action.shape),
            "mae": mae,
            "mse": mse,
            "gt_action": tensor_to_list(gt_action),
            "pred_action": tensor_to_list(pred_action),
        }
        records.append(rec)
        print(f"[batch {batch_idx}] mae={mae:.4f} mse={mse:.4f} gt_shape={tuple(gt_action.shape)} pred_shape={tuple(pred_action.shape)}")

        if args.save_frames:
            frames = obs["image"]  # (B, T, C, H, W)
            frames_path = args.out_dir / f"batch{batch_idx:04d}_frames.png"
            # Flatten time into grid row to see all frames
            frames_flat = frames.flatten(0, 1)
            save_image(frames_flat, str(frames_path), nrow=frames.shape[1])
            print(f"  saved input frames -> {frames_path}")

    with json_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} prediction records to {json_path}")


if __name__ == "__main__":
    main()
