#!/usr/bin/env python
#
# Minimal UVA training loop on LeRobot datasets (SO-101 by default).
# - Uses the UVA architecture from the neighbouring `unified_video_action` repo.
# - Consumes LeRobot v3 datasets via the UVALeRobotDataset adapter.
# - Runs in a plain venv (no Hydra/conda required).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from lerobot.uva_bridge.dataset import UVALeRobotDataset, _default_action_indices, _default_image_indices


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "lerobot").exists():
            return parent
    return Path.cwd()


def _add_uva_to_path(repo_root: Path) -> None:
    uva_root = repo_root.parent / "unified_video_action"
    if uva_root.exists():
        sys.path.append(str(uva_root))
    else:
        raise FileNotFoundError(
            f"Expected UVA repo at {uva_root}. Please clone it next to lerobot or set PYTHONPATH accordingly."
        )


def build_shape_meta(image_shape, action_dim, img_len, act_len):
    # Use OmegaConf DictConfig for attribute-style access expected by UVA
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


def load_uva_defaults(uva_root: Path):
    cfg_path = uva_root / "unified_video_action" / "config" / "model" / "uva.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not locate UVA default config at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    # We only reuse the model subsection; training loop is custom here
    return cfg.policy


def format_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    obs = {k: v.to(device=device, dtype=torch.float32) for k, v in batch["obs"].items()}
    action = batch["action"].to(device=device, dtype=torch.float32)
    return {"obs": obs, "action": action}


def main():
    parser = argparse.ArgumentParser(description="Train UVA on a LeRobot v3 dataset (SO-101 compatible).")
    parser.add_argument("--repo_id", required=True, help="Hugging Face dataset id or local path.")
    parser.add_argument("--root", default=None, help="Optional local cache/root for the dataset.")
    parser.add_argument("--image_key", default="observation.images.front", help="Camera feature key to use.")
    parser.add_argument("--action_key", default="action", help="Action feature key.")
    parser.add_argument(
        "--image_indices",
        nargs="+",
        type=int,
        default=None,
        help="Relative frame indices for images (default matches UVA: -12..16 step 4).",
    )
    parser.add_argument(
        "--action_indices",
        nargs="+",
        type=int,
        default=None,
        help="Relative frame indices for actions (default matches UVA: -15..16 step 1).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/uva_so101"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--betas", nargs=2, type=float, default=None, help="Adam betas, default from UVA config.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    _add_uva_to_path(repo_root)

    # Late imports after adjusting sys.path
    from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

    uva_root = repo_root.parent / "unified_video_action"
    policy_defaults = load_uva_defaults(uva_root)

    image_indices = args.image_indices if args.image_indices is not None else _default_image_indices()
    action_indices = args.action_indices if args.action_indices is not None else _default_action_indices()

    train_ds = UVALeRobotDataset(
        args.repo_id,
        root=args.root,
        image_key=args.image_key,
        action_key=args.action_key,
        image_delta_indices=image_indices,
        action_delta_indices=action_indices,
        split="train",
    )
    val_ds = UVALeRobotDataset(
        args.repo_id,
        root=args.root,
        image_key=args.image_key,
        action_key=args.action_key,
        image_delta_indices=image_indices,
        action_delta_indices=action_indices,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    shape_meta = build_shape_meta(
        image_shape=train_ds.image_shape,
        action_dim=train_ds.action_dim,
        img_len=len(image_indices),
        act_len=len(action_indices),
    )

    # Build UVA policy (images only, no proprioception/text for SO-101)
    policy_kwargs = {
        "vae_model_params": policy_defaults.vae_model_params,
        "autoregressive_model_params": policy_defaults.autoregressive_model_params,
        "action_model_params": {**policy_defaults.action_model_params, "predict_action": True},
        "shape_meta": shape_meta,
        "n_action_steps": policy_defaults.n_action_steps,
        "shift_action": True,
        "language_emb_model": None,
        # Use "umi" substring to reuse UVA image handling without extra keys.
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
    policy = UnifiedVideoActionPolicy(**policy_kwargs).to(args.device)

    betas = tuple(policy_defaults.optimizer.betas) if args.betas is None else tuple(args.betas)
    optimizer = policy.get_optimizer(
        weight_decay=args.weight_decay, learning_rate=args.lr, betas=betas  # type: ignore[arg-type]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and args.device.startswith("cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            batch = format_batch(batch, device=torch.device(args.device))
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, (loss_diff, loss_act) = policy(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.3f}", action=f"{loss_act:.3f}")

        avg_train = sum(train_losses) / max(1, len(train_losses))

        policy.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = format_batch(batch, device=torch.device(args.device))
                loss, _ = policy(batch)
                val_losses.append(loss.item())
        avg_val = sum(val_losses) / max(1, len(val_losses)) if val_losses else 0.0

        print(f"[epoch {epoch}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        ckpt_path = args.output_dir / f"epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "shape_meta": shape_meta,
                "image_indices": image_indices,
                "action_indices": action_indices,
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
