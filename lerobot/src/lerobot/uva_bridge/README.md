# UVA × LeRobot quick guide

This folder bridges the UVA model (from `../unified_video_action`) to LeRobot v3 datasets (e.g., SO‑101).

## Prerequisites
- `unified_video_action` repo cloned next to `lerobot/`  
  (`../unified_video_action/unified_video_action/...` is assumed).
- Pretrained UVA assets in that repo (KL‑VAE `pretrained_models/vae/kl16.ckpt`, MAR `pretrained_models/mar/...`) or adjust paths in the scripts.
- Python venv with `torch`, `omegaconf`, UVA deps, and LeRobot installed (`pip install -e .` inside `lerobot`).

## Train (action‑only by default)
```bash
cd lerobot
python src/lerobot/scripts/train_uva_so101.py \
  --repo_id <hf-dataset-or-local-path> \
  --image_key observation.images.front \
  --action_key action \
  --batch_size 8 --epochs 5 \
  --output_dir outputs/uva_so101
```
Notes:
- The script currently sets `task_modes=["policy_model"]`, so it trains only the action diffusion head. Video diffusion is **not** trained unless you change the modes (see below).
- Images are resized to 256x256 (UVA's expected resolution).
- Temporal windows: images at [-12, -8, -4, 0, 4, 8, 12, 16] (8 frames), actions at [-3..16] (20 frames → 16 after internal slicing). Override with `--image_indices/--action_indices` if your dataset requires.
- Checkpoints saved per epoch in `output_dir`.

## Enable video diffusion (optional)
- **Joint video+action**: edit `train_uva_so101.py` and set `task_modes=["video_model", "policy_model"]` (or remove the explicit `task_modes` override so the model uses UVA’s full mode list when `selected_training_mode` is `None`).
- **Video-only pretrain**: set `action_model_params["predict_action"]=False` and `task_modes=["video_model"]`. Train, then finetune jointly by restoring `predict_action=True`, pointing `autoregressive_model_params["pretrained_model_path"]` to the video checkpoint, and setting `task_modes` to include `"policy_model"`.

## Evaluate
```bash
python src/lerobot/scripts/eval_uva_so101.py \
  --checkpoint outputs/uva_so101/epoch0005.pt \
  --repo_id <hf-dataset-or-local-path>
```
Prints mean loss on the validation split (last 5% by default).

## Using proprioception/state (optional)
- Extend `UVALeRobotDataset` to include state keys from your LeRobot dataset and add them into `obs`.
- In `train_uva_so101.py`, set `use_proprioception=True`, add shapes to `shape_meta["obs"]`, and adjust `task_name` (contains "umi" to reuse UVA image handling).

## Troubleshooting
- If you see `FileNotFoundError` for UVA config/weights, ensure `../unified_video_action` exists or patch paths in the scripts.
- If `torch` import fails in your environment, verify GPU/CPU install; the scripts themselves are plain PyTorch without Hydra/conda. 
