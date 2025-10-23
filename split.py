#!/usr/bin/env python3
import argparse, random, shutil
from pathlib import Path

VIDEO_EXTS = {".mp4"}

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def split_counts(n, train_ratio, val_ratio):
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test

def place_unique(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        return dst
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1

def move_or_copy(src: Path, dst: Path, do_copy: bool):
    final_dst = place_unique(dst)
    if do_copy:
        shutil.copy2(src, final_dst)
    else:
        shutil.move(str(src), str(final_dst))

def main():
    ap = argparse.ArgumentParser(description="Split classed video folder into train/val/test.")
    ap.add_argument("--src", default="/data/Saniah/Video/Datasets/Dataset_30fps_cleaned", help="Folder containing class subfolders with videos")
    ap.add_argument("--out", default='/data/Saniah/Video/Datasets/Dataset_30fps_cleaned_split', help="Output base folder (default: create inside --src)")
    ap.add_argument("--train", type=float, default=0.80, help="Train ratio (default 0.70)")
    ap.add_argument("--val",   type=float, default=0.125, help="Val ratio (default 0.15)")
    ap.add_argument("--seed",  type=int,   default=42,   help="Random seed (default 42)")
    ap.add_argument("--move",  action="store_true",      help="Move files instead of copying")
    args = ap.parse_args()

    assert 0 < args.train < 1 and 0 <= args.val < 1 and args.train + args.val < 1, "Invalid ratios."

    src = Path(args.src).resolve()
    out = Path(args.out).resolve() if args.out else src
    rng = random.Random(args.seed)

    # Class folders = immediate subdirectories of src
    class_dirs = [d for d in src.iterdir() if d.is_dir() and d.name.lower() not in {"train","val","test"}]
    if not class_dirs:
        raise SystemExit("No class subfolders found in --src.")

    totals = {"train":0, "val":0, "test":0}
    for cls_dir in sorted(class_dirs):
        vids = [p for p in cls_dir.iterdir() if is_video(p)]
        if not vids:
            print(f"[WARN] No videos in {cls_dir.name}, skipping.")
            continue

        rng.shuffle(vids)
        n_train, n_val, n_test = split_counts(len(vids), args.train, args.val)
        splits = {
            "train": vids[:n_train],
            "val":   vids[n_train:n_train+n_val],
            "test":  vids[n_train+n_val:],
        }

        for split, files in splits.items():
            for f in files:
                dst = out / split / cls_dir.name / f.name
                move_or_copy(f, dst, do_copy=not args.move)
            totals[split] += len(files)

        print(f"{cls_dir.name}: train {n_train}, val {n_val}, test {n_test}")

    mode = "MOVE" if args.move else "COPY"
    print(f"\nDone. Mode: {mode}")
    print(f"Totals -> Train: {totals['train']} | Val: {totals['val']} | Test: {totals['test']}")
    print(f"Output base: {out}")

if __name__ == "__main__":
    main()
