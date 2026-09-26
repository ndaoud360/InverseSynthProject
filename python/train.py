#!/usr/bin/env python3
"""
Phase 3 / Stage 1 training.

Variable-length batching: examples are grouped into buckets of similar
length so each batch is padded only to its own max length (typically <5 %
padding), then the model's masks ignore the padding.

    python train.py --train data/train --val data/val --epochs 40 --bs 64
"""
import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from invsynth import params as P
from invsynth.model import PAD_VALUE, InverseSynthNet, param_loss


class ShardDataset(Dataset):
    def __init__(self, root):
        self.shards = []
        self.index = []
        for si, d in enumerate(sorted(glob.glob(os.path.join(root, "shard_*")))):
            sh = dict(mels=np.load(os.path.join(d, "mels.npy"), mmap_mode="r"),
                      off=np.load(os.path.join(d, "offsets.npy")),
                      ctx=np.load(os.path.join(d, "ctx.npy")),
                      y=np.load(os.path.join(d, "params.npy")))
            self.shards.append(sh)
            self.index += [(si, i) for i in range(len(sh["y"]))]
        if not self.index:
            raise RuntimeError(f"no shards in {root}")
        self.lengths = np.array([self.shards[s]["off"][i + 1] - self.shards[s]["off"][i]
                                 for s, i in self.index])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        s, i = self.index[k]
        sh = self.shards[s]
        a, b = sh["off"][i], sh["off"][i + 1]
        mel = torch.from_numpy(np.asarray(sh["mels"][:, a:b], dtype=np.float32))
        y = sh["y"][i]
        rel = P.relevance_mask(y)
        return mel, torch.from_numpy(sh["ctx"][i]), torch.from_numpy(y), torch.from_numpy(rel)


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=50):
        self.lengths, self.bs, self.shuffle, self.mult = lengths, batch_size, shuffle, bucket_mult

    def __iter__(self):
        idx = np.random.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        chunk = self.bs * self.mult
        for c in range(0, len(idx), chunk):
            part = idx[c:c + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[j:j + self.bs].tolist() for j in range(0, len(part), self.bs)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)


def collate(batch):
    mels, ctx, y, rel = zip(*batch)
    lengths = torch.tensor([m.shape[1] for m in mels])
    T = int(lengths.max())
    out = torch.full((len(mels), mels[0].shape[0], T), PAD_VALUE)
    for i, m in enumerate(mels):
        out[i, :, :m.shape[1]] = m
    return out, torch.stack(ctx), torch.stack(y), torch.stack(rel), lengths


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot, n, mae = 0.0, 0, torch.zeros(P.N_PARAMS)
    from invsynth.model import assemble_vector
    for mel, ctx, y, rel, L in loader:
        mel, ctx, y, rel, L = mel.to(dev), ctx.to(dev), y.to(dev), rel.to(dev), L.to(dev)
        cont, cats = model(mel, ctx, L)
        loss, _, _ = param_loss(cont, cats, y, rel)
        tot += loss.item() * len(y); n += len(y)
        mae += ((assemble_vector(cont, cats) - y).abs() * rel).sum(0).cpu()
    return tot / n, mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    dev = pick_device()
    tr, va = ShardDataset(a.train), ShardDataset(a.val)
    tl = DataLoader(tr, batch_sampler=BucketBatchSampler(tr.lengths, a.bs), collate_fn=collate,
                    num_workers=a.workers, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_sampler=BucketBatchSampler(va.lengths, a.bs, shuffle=False),
                    collate_fn=collate, num_workers=a.workers)
    model = InverseSynthNet().to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {dev}, "
          f"{len(tr)} train / {len(va)} val")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl),
                                                pct_start=0.1)
    use_amp = dev.type == "cuda"
    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, (mel, ctx, y, rel, L) in enumerate(tl):
            mel, ctx, y, rel, L = mel.to(dev), ctx.to(dev), y.to(dev), rel.to(dev), L.to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                cont, cats = model(mel, ctx, L)
            loss, lc, lk = param_loss(cont.float(), [c.float() for c in cats], y, rel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{len(tl)} loss {loss.item():.4f} "
                      f"(cont {lc.item():.4f} cat {lk.item():.4f})")
        vloss, mae = evaluate(model, vl, dev)
        print(f"== epoch {ep}: train {run / len(tl):.4f}  val {vloss:.4f}  "
              f"({time.time() - t0:.0f}s)")
        worst = torch.argsort(mae, descending=True)[:5]
        print("   worst params:", ", ".join(f"{P.PARAMS[i].name}={mae[i]:.3f}" for i in worst))
        ckpt = dict(model=model.state_dict(), epoch=ep, val=vloss)
        torch.save(ckpt, os.path.join(a.out, "last.pt"))
        if vloss < best:
            best = vloss
            torch.save(ckpt, os.path.join(a.out, "best.pt"))


if __name__ == "__main__":
    main()
