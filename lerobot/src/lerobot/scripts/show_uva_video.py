#!/usr/bin/env python
#
# Sample predicted future video frames from a UVA checkpoint and save a side-by-side grid
# of input (conditioning) frames vs. predicted frames.
#
# Example:
#   python src/lerobot/scripts/show_uva_video.py \
#       --checkpoint outputs/uva_so101_video/epoch0002.pt \
#       --repo_id tenkau/record-test \
#       --image_key observation.images.laptop \
#       --out_img uva_outputs/video_pred_epoch2.png

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
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


def main():
    parser = argparse.ArgumentParser(description="Visualize UVA predicted video frames.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image_key", default="observation.images.front")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--image_indices", nargs="+", type=int, default=None)
    parser.add_argument("--action_indices", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_img", type=Path, default=Path("uva_outputs/video_pred.png"))
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to sample from.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index within the chosen split to visualize.")
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
    from unified_video_action.utils.data_utils import extract_latent_autoregressive

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
        split=args.split,
    )
    if args.sample_index >= len(val_ds):
        raise IndexError(f"sample_index {args.sample_index} out of range for val split of length {len(val_ds)}")

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
        "task_modes": ["video_model", "policy_model"],
        "normalizer_type": "none",
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
    # Fetch the requested sample directly
    sample = val_ds[args.sample_index]
    obs = {k: v.unsqueeze(0).to(device) for k, v in sample["obs"].items()}
    # Preprocess to VAE latent space (match training/inference pipeline)
    imgs = obs["image"] * 255.0  # (B, T, C, H, W)
    imgs = imgs.permute(0, 2, 1, 3, 4) / 127.5 - 1.0  # to (B, C, T, H, W) in [-1, 1]
    cond_latent, _ = extract_latent_autoregressive(policy.vae_model, imgs)
    cond_latent = cond_latent[:, : policy.model.n_frames]  # keep the conditioning window

    with torch.no_grad():
        tokens, _ = policy.model.sample_tokens(
            bsz=1,
            cond=cond_latent,
            text_latents=None,
            num_iter=policy.autoregressive_model_params.num_iter,
            cfg=policy.autoregressive_model_params.cfg,
            cfg_schedule=policy.autoregressive_model_params.cfg_schedule,
            temperature=policy.autoregressive_model_params.temperature,
            history_nactions=None,
            nactions=None,
            proprioception_input={},
            task_mode="video_model",
            vae_model=policy.vae_model,
        )

    # Decode VAE latents back to RGB
    decoded = policy.vae_model.decode(tokens)
    decoded = (decoded.clamp(-1, 1) + 1) / 2.0  # to [0,1]
    n_frames = policy.model.n_frames
    decoded = decoded.view(1, n_frames, *decoded.shape[1:])

    cond_frames = obs["image"][:, :n_frames].detach().cpu().clamp(0, 1)
    pred_frames = decoded.detach().cpu().clamp(0, 1)

    cond_flat = cond_frames.flatten(0, 1)
    pred_flat = pred_frames.flatten(0, 1)
    grid = torch.cat([cond_flat, pred_flat], dim=0)

    args.out_img.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(args.out_img), nrow=n_frames, padding=2)
    print(f"Saved grid with {n_frames} conditioning frames (top row) and {n_frames} predicted frames (bottom row) -> {args.out_img}")


if __name__ == "__main__":
    main()
