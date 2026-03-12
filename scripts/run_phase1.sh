#!/bin/bash
# ============================================================
# Phase 1 Experiments — Baseline vs OT-CellFlux
# Run these from the CellFlux repo root on your Jupyter server
# ============================================================

# ── 0. Smoke test (1 batch, fast, no GPU required) ──────────
# Run this first to confirm OT code doesn't crash
python3 train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=1 \
    --epochs=1 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --use_initial=2 \
    --noise_level=1.0 \
    --noise_prob=0.5 \
    --use_ot_pairing \
    --ot_feature_space=pooled_mean_std \
    --ot_cost=cosine \
    --ot_hard_method=hungarian \
    --test_run \
    --output_dir=outputs/ot_smoke

echo "Smoke test done. Check outputs/ot_smoke for logs."
echo ""
echo "If smoke test passes, run the full experiments below."

# ── 1. Baseline (matches original CellFlux paper exactly) ───
python3 train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=1 \
    --eval_frequency=25 \
    --epochs=200 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=5120 \
    --use_initial=2 \
    --noise_level=1.0 \
    --noise_prob=0.5 \
    --save_fid_samples \
    --output_dir=outputs/baseline_200ep

# ── 2. OT-CellFlux (corrected, recommended settings) ────────
python3 train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=1 \
    --eval_frequency=25 \
    --epochs=200 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=5120 \
    --use_initial=2 \
    --noise_level=1.0 \
    --noise_prob=0.5 \
    --save_fid_samples \
    --use_ot_pairing \
    --ot_feature_space=pooled_mean_std \
    --ot_cost=cosine \
    --ot_hard_method=hungarian \
    --output_dir=outputs/ot_v2_hungarian_200ep

# ── 3. (Optional) Quick ablation: 50 epochs for fast comparison
# python train.py \
#     --dataset=bbbc021 \
#     --config=bbbc021_all \
#     --batch_size=32 \
#     --accum_iter=1 \
#     --eval_frequency=10 \
#     --epochs=50 \
#     --class_drop_prob=0.2 \
#     --cfg_scale=0.2 \
#     --compute_fid \
#     --ode_method heun2 \
#     --ode_options '{"nfe": 50}' \
#     --use_ema \
#     --edm_schedule \
#     --skewed_timesteps \
#     --fid_samples=5120 \
#     --use_initial=2 \
#     --noise_level=1.0 \
#     --noise_prob=0.5 \
#     --save_fid_samples \
#     --use_ot_pairing \
#     --ot_feature_space=pooled_mean_std \
#     --ot_cost=cosine \
#     --ot_hard_method=hungarian \
#     --output_dir=outputs/ot_v2_50ep_ablation

# ── 4. Evaluate FID + KID after training ────────────────────
# python eval_fid.py \
#     --model_name baseline_200ep \
#     --dataset bbbc021_all \
#     --image_root outputs/baseline_200ep/fid_samples/epoch-200
#
# python eval_fid.py \
#     --model_name ot_v2_200ep \
#     --dataset bbbc021_all \
#     --image_root outputs/ot_v2_hungarian_200ep/fid_samples/epoch-200
