"""Entry point.

    python main.py baseline --model eegnet --folds dev          # 6 dev folds
    python main.py baseline --model eegnet --folds dev --only dev0 --subjects 1-12   # quick smoke test
    python main.py baseline --model eegnet --folds loso --repeats 5                  # final run

Results are written to results/<run name>/per_subject.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eegproj.data import PreprocConfig, load_subjects
from eegproj.folds import ALL_SUBJECTS, ChannelScaler, dev_folds, loso_folds
from eegproj.models import build_model
from eegproj.train import TrainConfig, TrialDataset, fit, get_device, predict


def parse_subjects(spec: str | None) -> list[int]:
    if not spec:
        return ALL_SUBJECTS
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def stack(data: dict, subjects: list[int], session: str):
    X = np.concatenate([data[s][session]["X"] for s in subjects])
    y = np.concatenate([data[s][session]["y"] for s in subjects])
    sub = np.concatenate([np.full(len(data[s][session]["y"]), s) for s in subjects])
    return X, y, sub


def run_baseline(args):
    subjects = parse_subjects(args.subjects)
    pre = PreprocConfig(car=args.car)
    n_val = max(1, min(5, len(subjects) // 6))
    folds = (dev_folds(subjects, n_val=n_val) if args.folds == "dev" else loso_folds(subjects, n_val=n_val))
    if args.only:
        folds = [f for f in folds if f.name in args.only.split(",")]

    print(f"Loading {len(subjects)} subjects (first run downloads and caches them)...")
    data = load_subjects(subjects, pre, args.cache_dir)

    device = get_device()
    run_name = args.name or f"{args.model}_{args.folds}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "preproc": pre.__dict__}, indent=2))

    rows = []
    for rep in range(args.repeats):
        for i, fold in enumerate(folds):
            seed = 1000 * rep + i
            Xtr, ytr, str_ = stack(data, fold.train, "1")
            Xva, yva, sva = stack(data, fold.val, "2")
            Xte, yte, ste = stack(data, fold.test, "2")
            scaler = ChannelScaler().fit(Xtr)
            tr = TrialDataset(scaler.transform(Xtr), ytr, str_)
            va = TrialDataset(scaler.transform(Xva), yva, sva)
            te = TrialDataset(scaler.transform(Xte), yte, ste)

            torch.manual_seed(seed)
            model = build_model(args.model, n_chans=Xtr.shape[1], n_times=Xtr.shape[2], sfreq=pre.sfreq)
            cfg = TrainConfig(epochs=args.epochs, lr=args.lr, patience=args.patience, seed=seed)
            print(f"[rep {rep}] {fold.name}: train {len(fold.train)} people / {len(ytr)} trials, "
                  f"val {fold.val}, test {fold.test}  on {device}")
            model, info = fit(model, tr, va, cfg, device)

            p, y, s = predict(model, torch.utils.data.DataLoader(te, batch_size=256), device)
            for subj in np.unique(s):
                m = s == subj
                rows.append({"repeat": rep, "fold": fold.name, "subject": int(subj),
                             "acc": float((p[m] == y[m]).mean()), "n_trials": int(m.sum()),
                             "best_epoch": info["best"]["epoch"], "val_acc": info["best"]["val_acc"]})
            fold_acc = np.mean([r["acc"] for r in rows if r["fold"] == fold.name and r["repeat"] == rep])
            print(f"  -> test acc {fold_acc:.3f} (best epoch {info['best']['epoch']})")
            pd.DataFrame(rows).to_csv(out_dir / "per_subject.csv", index=False)

    df = pd.DataFrame(rows)
    per_subject = df.groupby("subject")["acc"].mean()
    print(f"\n{args.model}: mean {per_subject.mean():.4f} ± {per_subject.std():.4f} "
          f"over {len(per_subject)} unseen people  ->  {out_dir}/per_subject.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="Stage 1: cross-subject baseline")
    b.add_argument("--model", default="eegnet", choices=["eegnet", "shallow"])
    b.add_argument("--folds", default="dev", choices=["dev", "loso"])
    b.add_argument("--only", help="comma-separated fold names, e.g. dev0")
    b.add_argument("--subjects", help="e.g. 1-12 or 1,3,5-9 (default: all 54)")
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--epochs", type=int, default=100)
    b.add_argument("--patience", type=int, default=20)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--car", action="store_true", help="common average reference")
    b.add_argument("--cache-dir", default="cache")
    b.add_argument("--name")
    args = ap.parse_args()
    if args.cmd == "baseline":
        run_baseline(args)


if __name__ == "__main__":
    main()
