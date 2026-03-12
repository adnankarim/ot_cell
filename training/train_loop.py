# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
import random
import time
from typing import Iterable
from training.dataloader import CellDataLoader
from training.feature_utils import get_ot_features
import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def my_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    datamodule: CellDataLoader,
    use_initial: int,
    ot_matcher=None,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    for data_iter_step, batch in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break
        
        x_real, y_trg, y_mod = batch['X'], batch['mols'], batch['y_id']
        x_real_ctrl, x_real_trt = x_real
        x_real_ctrl, x_real_trt = x_real_ctrl.to(device), x_real_trt.to(device)

        # Move labels to device early so OT can reorder them if needed
        y_trg = y_trg.long().to(device)
        if torch.is_tensor(y_mod):
            y_mod = y_mod.long().to(device)

        # ── Batch-constrained OT pairing ──────────────────────────────────────
        # Correct design:
        #   1. Reorder TREATED images AND their labels together — controls are fixed.
        #      This keeps y_trg[i] aligned with x_real_trt[i] after re-matching.
        #   2. Run OT within each experimental batch only (plate/batch id) to
        #      respect CellFlux's same-batch biological assumption.
        #   3. Use global permutation mapping so scatter-writes don't corrupt indices.
        if ot_matcher is not None:
            ot_mode = getattr(args, "ot_feature_space", "pooled_mean_std")

            batch_ids = batch.get("batch", None)
            if torch.is_tensor(batch_ids):
                batch_ids_np = batch_ids.detach().cpu().numpy()
            else:
                batch_ids_np = np.array(batch_ids)

            # Start with identity permutation; fill in OT-optimal per-batch groups
            global_perm = torch.arange(x_real_trt.size(0), device=device)

            for b in np.unique(batch_ids_np):
                group_idx_np = np.where(batch_ids_np == b)[0]
                if len(group_idx_np) <= 1:
                    continue  # OT undefined for singleton groups

                group_idx = torch.as_tensor(group_idx_np, device=device, dtype=torch.long)
                ctrl_sub = x_real_ctrl[group_idx]
                trt_sub  = x_real_trt[group_idx]

                ctrl_feat = get_ot_features(ctrl_sub, mode=ot_mode)
                pert_feat = get_ot_features(trt_sub,  mode=ot_mode)

                # perm[i] = which treated index to assign to ctrl[i]
                local_perm = ot_matcher.get_indices(ctrl_feat, pert_feat).to(device)

                # Map local indices back to global batch positions
                global_perm[group_idx] = group_idx[local_perm]

            # Reorder treated images AND their labels consistently
            x_real_trt = x_real_trt[global_perm]
            y_trg      = y_trg[global_perm]
            if torch.is_tensor(y_mod):
                y_mod = y_mod[global_perm]
        # ──────────────────────────────────────────────────────────────────────

        y_org = None
        z_emb_trg = datamodule.embedding_matrix(y_trg).to(device)
        samples = None

        labels = None
        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"concat_conditioning": z_emb_trg}
        
        if args.discrete_flow_matching:
            samples = (samples * 255.0).to(torch.long)
            t = torch.torch.rand(samples.shape[0]).to(device)
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            )
            path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            logits = model(path_sample.x_t, t=t, extra=conditioning)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), samples.reshape([-1])
            ).mean()
        else:
            if args.skewed_timesteps:
                t = skewed_timestep_sample(x_real_ctrl.shape[0], device=device)
            else:
                t = torch.torch.rand(x_real_ctrl.shape[0]).to(device)
            if use_initial == 1:
                x_0 = x_real_ctrl
            elif use_initial == 2:
                p_r = random.random()
                if p_r > args.noise_prob:
                    x_0 = x_real_ctrl
                else:
                    x_0 = x_real_ctrl + torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device) * args.noise_level
            else:
                x_0 = torch.randn(x_real_ctrl.shape, dtype=torch.float32, device=device)
            
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_real_trt)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
