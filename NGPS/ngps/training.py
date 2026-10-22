from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.manifest import load_manifest, manifest_hash, noisy_items
from datasets.triplet_dataset import VolumeTripletDataset
from .baselines import adjacent_loss, blindspot_loss
from .config import load_config
from .guide import make_misalignment_mask
from .losses import NGPSHybridLoss, NSN2NLoss
from .nafnet import build_nafnet
from .provenance import provenance, write_json
from .sampler import NeighborGuidedPatchSampler
from .seed import seed_everything


def mask_diagnostics(prev, next_):
    masks = torch.cat([prev, next_], dim=0)
    active = masks.flatten(1).mean(1)
    return {"dynamic_active_fraction": float(active.mean()),
            "dynamic_empty_fraction": float((active == 0).float().mean()),
            "static_active_fraction": float(1 - active.mean()),
            "static_empty_fraction": float((active == 1).float().mean())}


def train(manifest_path, config_path, output_dir, device="cpu", sigma=0.06,
          method="ngps", central_fraction=1.0, save_every=5):
    cfg, train_cfg, raw_cfg = load_config(config_path)
    if method not in {"ngps", "nsn2n", "adjacent", "blindspot"} or save_every < 1:
        raise ValueError("Invalid method or save interval")
    seed_everything(train_cfg.seed)
    manifest = load_manifest(manifest_path, prepared=True)
    device = torch.device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run: {out}")
    # Neither dataset receives a clean path. Validation is full-slice, no crops.
    datasets = {split: VolumeTripletDataset(
        noisy_items(manifest, manifest_path, split, sigma), cfg,
        crop_size=train_cfg.crop_size if split == "train" else 0,
        central_fraction=central_fraction if split == "train" else 1.0,
        use_guides=method in {"ngps", "nsn2n"} and split == "train",
        seed=train_cfg.seed,
    ) for split in ("train", "val")}
    loader = DataLoader(datasets["train"], batch_size=train_cfg.batch_size, shuffle=True,
                        num_workers=train_cfg.num_workers,
                        generator=torch.Generator().manual_seed(train_cfg.seed))
    val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=0)
    model = build_nafnet(cfg).to(device)
    sampler = NeighborGuidedPatchSampler(cfg).to(device)
    hybrid = NGPSHybridLoss(cfg.rc_weight)
    nsn2n = NSN2NLoss(cfg.rc_weight, float(raw_cfg.get("nsn2n", {}).get("lambda_ic", 1.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    mask_rng = torch.Generator().manual_seed(train_cfg.seed)
    metadata = {**provenance(device, source_snapshot=out / "source_snapshot.zip"),
                "schema_version": 1, "architecture_version": 2,
                "config": raw_cfg, "ngps_config": asdict(cfg), "training_config": asdict(train_cfg),
                "seed": train_cfg.seed, "manifest": manifest, "manifest_hash": manifest_hash(manifest),
                "method": method, "sigma": sigma, "central_fraction": central_fraction,
                "selection": {"rule": "fixed_final_epoch", "epoch": train_cfg.epochs,
                              "clean_references_used": False},
                "validation_protocol": "fixed random masked noisy pixels, all interior val slices; monitor only",
                "validation_history": [], "training_history": [], "epoch": 0}
    write_json(out / "config.json", raw_cfg)
    write_json(out / "manifest.json", manifest)
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        records = []
        for batch in loader:
            center, prev, next_, gc, gp, gn = [x.to(device) for x in batch]
            diagnostics = {}
            if method == "ngps":
                targets = sampler(gc, gp, gn, prev, next_)
                loss = hybrid(model(center), prev, next_, targets.target_prev, targets.target_next,
                              targets.mask_prev, targets.mask_next, model(prev), model(next_)).total
                diagnostics = mask_diagnostics(targets.mask_prev, targets.mask_next)
                diagnostics["match_cost_mean"] = float((targets.cost_prev.mean() + targets.cost_next.mean()) / 2)
                diagnostics["match_cost_max"] = float(torch.maximum(targets.cost_prev.max(), targets.cost_next.max()))
            elif method == "nsn2n":
                mp = make_misalignment_mask(gc, gp, cfg.mask_threshold)
                mn = make_misalignment_mask(gc, gn, cfg.mask_threshold)
                loss = nsn2n(model, center, prev, next_, mp, mn).total
                diagnostics = mask_diagnostics(mp, mn)
            elif method == "adjacent":
                loss = adjacent_loss(model, center, prev, next_)
            else:
                loss = blindspot_loss(model, center, mask_rng)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            records.append({"loss": float(loss.detach()), **diagnostics})
        model.eval()
        val_rng = torch.Generator().manual_seed(train_cfg.seed + 1)
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(blindspot_loss(model, batch[0].to(device), val_rng)))
        val_loss = float(np.mean(val_losses))
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite noisy validation loss")
        metadata["epoch"] = epoch
        metadata["training_history"].append({"epoch": epoch, **{
            key: float((np.max if key == "match_cost_max" else np.mean)([r[key] for r in records]))
            for key in records[0]}})
        metadata["validation_history"].append({"epoch": epoch, "noisy_masked_mse": val_loss})
        write_json(out / "metadata.json", metadata)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "ngps_config": asdict(cfg), "training_config": asdict(train_cfg), "metadata": metadata}
        if epoch % save_every == 0 or epoch == train_cfg.epochs:
            torch.save(state, out / f"model_E{epoch}.pth")
        print({**metadata["training_history"][-1], "noisy_validation_mse": val_loss})
    torch.save(state, out / "model_final.pth")
    return out / "model_final.pth"


def main(default_method="ngps"):
    parser = argparse.ArgumentParser(description="Manifest-based noisy-only training; fixed final-epoch selection")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sigma", type=float, choices=[0.03, 0.06, 0.10], default=0.06)
    parser.add_argument("--method", choices=["ngps", "nsn2n", "adjacent", "blindspot"], default=default_method)
    parser.add_argument("--config", default=f"configs/{default_method}_nafnet_default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--central-fraction", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    print(train(args.manifest, args.config, args.output_dir, args.device, args.sigma,
                args.method, args.central_fraction, args.save_every))
