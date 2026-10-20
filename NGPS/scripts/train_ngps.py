from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_volume_items
from ngps import NGPSHybridLoss, NeighborGuidedPatchSampler, build_nafnet, load_config
from ngps.run_metadata import RunRecorder
from ngps.seed import seed_everything


def _loss_for_batch(model, sampler, criterion, batch, device: torch.device):
    raw_center, raw_prev, raw_next, guide_center, guide_prev, guide_next = (tensor.to(device) for tensor in batch)
    with torch.no_grad():
        targets = sampler(guide_center, guide_prev, guide_next, raw_prev, raw_next)
    out_center = model(raw_center)
    out_prev = model(raw_prev)
    out_next = model(raw_next)
    return criterion(
        out_center,
        raw_prev,
        raw_next,
        targets.target_prev,
        targets.target_next,
        targets.mask_prev,
        targets.mask_next,
        out_prev,
        out_next,
    )


def _mean_validation_loss(model, sampler, criterion, loader, device: torch.device, max_batches: int | None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            losses.append(float(_loss_for_batch(model, sampler, criterion, batch, device).total.cpu()))
    if not losses:
        raise RuntimeError("Validation loader yielded no batches.")
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NGPS using noisy-only train/validation volumes.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/ngps_nafnet_default.yaml")
    parser.add_argument("--output-dir", default="runs/ngps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--central-fraction", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test control; recorded in run metadata.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Smoke-test control; recorded in run metadata.")
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        parser.error("--max-val-batches must be positive")

    cfg, train_cfg, raw_cfg = load_config(args.config)
    seed_everything(train_cfg.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded_config = {
        **raw_cfg,
        "runtime": {
            "central_fraction": args.central_fraction,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "validation_selection": "minimum noisy-only NGPS self-supervised validation objective",
        },
    }
    recorder = RunRecorder(
        out_dir / "run_manifest.json",
        command=sys.argv,
        config=recorded_config,
        seed=train_cfg.seed,
        device=str(device),
        dataset_manifest_path=args.dataset_manifest,
    )

    try:
        train_items = list_volume_items(args.data_root, "train")
        val_items = list_volume_items(args.data_root, "val")
        train_dataset = VolumeTripletDataset(
            train_items,
            cfg,
            crop_size=train_cfg.crop_size,
            central_fraction=args.central_fraction,
            seed=train_cfg.seed,
        )
        val_dataset = VolumeTripletDataset(
            val_items,
            cfg,
            crop_size=train_cfg.crop_size,
            central_fraction=args.central_fraction,
            seed=train_cfg.seed + 1,
        )
        generator = torch.Generator().manual_seed(train_cfg.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
        )

        model = build_nafnet(cfg).to(device)
        sampler = NeighborGuidedPatchSampler(cfg).to(device).eval()
        criterion = NGPSHybridLoss(lambda_rc=cfg.rc_weight)
        if args.dry_run:
            train_dataset[0]
            val_dataset[0]
            recorder.finish(
                "dry_run",
                extra={
                    "train_subjects": [item.case_id for item in train_items],
                    "val_subjects": [item.case_id for item in val_items],
                    "train_sample_count": len(train_dataset),
                    "val_sample_count": len(val_dataset),
                    "clean_references_accessed": False,
                    "data_order_seed": train_cfg.seed,
                },
            )
            print(f"Dry run validated noisy-only data and config; wrote {out_dir / 'run_manifest.json'}")
            return

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        best_loss = float("inf")
        best_epoch = None
        validation_log = out_dir / "validation.jsonl"
        for epoch in range(1, train_cfg.epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            losses = []
            loop = tqdm(train_loader, desc=f"NGPS epoch {epoch}/{train_cfg.epochs}")
            for batch_index, batch in enumerate(loop):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                loss_out = _loss_for_batch(model, sampler, criterion, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss_out.total.backward()
                optimizer.step()
                losses.append(float(loss_out.total.detach().cpu()))
                loop.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")
            if not losses:
                raise RuntimeError("Training loader yielded no batches.")

            val_loss = _mean_validation_loss(model, sampler, criterion, val_loader, device, args.max_val_batches)
            checkpoint = {
                "model": model.state_dict(),
                "ngps_config": asdict(cfg),
                "training_config": asdict(train_cfg),
                "epoch": epoch,
                "validation_loss": val_loss,
                "selection_rule": "minimum noisy-only NGPS self-supervised validation objective",
            }
            torch.save(checkpoint, out_dir / "model_last.pth")
            if epoch % args.save_every == 0:
                torch.save(checkpoint, out_dir / f"model_E{epoch}.pth")
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(checkpoint, out_dir / "model_best.pth")
            with validation_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": sum(losses) / len(losses),
                            "validation_loss": val_loss,
                            "validation_uses_clean_references": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        recorder.finish(
            "complete",
            extra={
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "checkpoint_selection_rule": "minimum noisy-only NGPS self-supervised validation objective",
                "clean_references_accessed": False,
                "data_order_seed": train_cfg.seed,
            },
        )
        print(f"Saved last and best checkpoints to {out_dir}")
    except Exception as exc:
        recorder.finish("failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
