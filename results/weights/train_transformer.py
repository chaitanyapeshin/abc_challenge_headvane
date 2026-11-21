#!/usr/bin/env python
"""
train_transformer.py — 9th ABC Challenge: train, benchmark and package a TRANSFORMER head-pose predictor.

    python train_transformer.py --team MYTEAM --data data/headvane --smoke                 # quick check first
    python train_transformer.py --team MYTEAM --data data/headvane --device cuda:0 --aux 0.1
    python train_transformer.py ... --d-model 96 --blocks 4 --heads 4 --patch 5 --context 200   # variants

Model: transformer encoder with a motion-aware input embedding
  1. per-step features, all independent of the player's heading: angular velocity ω (3), head tilt (3),
     speed |ω| (1), angular acceleration Δω (3);
  2. patch embedding: a strided Conv1d turns every `patch` samples (default 5 = 50 ms) into one token;
  3. learned recency (positional) embedding per patch;
  4. a readout token, initialised from the newest sample, appended after the history; its output feeds
     the forecast head (20 rotations relative to the last orientation, on top of a learned
     constant-velocity guess).
Attention is written by hand (MatMul + Softmax) so the ONNX export is clean.

Same pipeline and data cache as train_seq.py / train_kalmannet.py: every user, user-disjoint 80/10/10 split,
challenge-metric loss, checkpoint on validation, TEST once, TensorBoard, ONNX export + checks, tested ZIP.
Output folder (default runs/transformer) is picked up by compare_all.py.
"""
import argparse, hashlib, json, math, os, random, shutil, subprocess, sys, textwrap, time, zipfile
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FS, DT = 100, 0.01
CONTEXT, HORIZON = 400, 20                       # challenge: 400 samples in, 20 out (200 ms)
DATASET = "chaitanyapeshin/headvane-boxrr23-preprocessed"


# ════════════════════════════════════ 1. data ════════════════════════════════════
VAL_FRACTION, TEST_FRACTION = 0.10, 0.10
SPLITS = ("train", "val", "test")


def split_of(user):
    """User-level split by a stable hash: every user goes to exactly one split.
    val = the same users as in the EDA notebook (h < 0.1)."""
    h = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) / 16 ** 8
    return "val" if h < VAL_FRACTION else "test" if h < VAL_FRACTION + TEST_FRACTION else "train"


def find_segments(f):
    groups = []
    f.visititems(lambda n, o: groups.append(o) if isinstance(o, h5py.Group) and "quaternions" in o else None)
    return groups


def cut_file(job):
    """Random non-overlapping pieces from every segment of one file → [(user, split, (n, L, 4))]."""
    path, seed, train_len, caps = job
    rng, out = np.random.default_rng(seed), []
    try:
        with h5py.File(path, "r") as f:
            for g in find_segments(f):
                user = str(g.attrs.get("user_id", Path(path).parent.name))
                split = split_of(user)
                L = train_len if split == "train" else CONTEXT + HORIZON
                q = g["quaternions"][:].astype(np.float32)          # (N, 4) w,x,y,z
                n = len(q) // L
                if n == 0:
                    continue
                start = rng.integers(0, len(q) - n * L + 1)
                pieces = q[start:start + n * L].reshape(n, L, 4)
                out.append((user, split, pieces[rng.permutation(n)[:caps[split]]]))
    except Exception as e:
        print(f"  skip {path}: {e}", flush=True)
    return out


def build_cache(files, cache, train_len, caps, workers, seed):
    """One parallel pass over all files → train.bin / val.bin / test.bin + meta.json (reused if present)."""
    meta_path = cache / "meta.json"
    if meta_path.exists():
        print(f"using cache {cache}")
        return json.loads(meta_path.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    count = {s: {} for s in SPLITS}
    users, owner = {}, {s: [] for s in SPLITS}
    fh = {s: open(cache / f"{s}.bin", "wb") for s in SPLITS}
    order = random.Random(seed).sample(files, len(files))
    jobs = [(str(p), seed + i, train_len, caps) for i, p in enumerate(order)]
    t0 = time.time()
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(cut_file, jobs, chunksize=4), 1):
            for user, split, pieces in res:
                room = caps[split] - count[split].get(user, 0)
                if room <= 0:
                    continue
                take = np.ascontiguousarray(pieces[:room])
                fh[split].write(take.tobytes())
                count[split][user] = count[split].get(user, 0) + len(take)
                owner[split] += [users.setdefault(user, len(users))] * len(take)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,} files ({time.time() - t0:.0f}s) | " +
                      " | ".join(f"{s} {len(owner[s]):,}" for s in SPLITS), flush=True)
    for f in fh.values():
        f.close()
    meta = {"train_len": train_len, "users": list(users)}
    for s in SPLITS:
        meta[f"n_{s}"], meta[f"{s}_users"] = len(owner[s]), len(count[s])
        meta[f"owner_{s}"] = owner[s]
    assert not (set(count["train"]) & set(count["val"]) or set(count["train"]) & set(count["test"])
                or set(count["val"]) & set(count["test"])), "user overlap between splits"
    meta_path.write_text(json.dumps(meta))
    return meta


def find_root(base):
    for pattern in ["users", "*/users", "*/*/users", "*/*/*/users", "*/*/*/*/users", "*/*/*/*/*/users"]:
        hit = next(Path(base).glob(pattern), None)
        if hit is not None and hit.is_dir():
            return hit.parent
    return None


def connect(args):
    if args.data:
        root = find_root(args.data)
        assert root, f"no users/ folder under {args.data}"
        return root
    if Path("/kaggle/input").exists() and (root := find_root("/kaggle/input")):
        return root
    if Path(args.download_dir).exists() and (root := find_root(args.download_dir)):
        return root
    import kagglehub
    try:
        kagglehub.whoami()
    except Exception:
        kagglehub.login()
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)
    print("downloading the corpus (~26 GB, first run only) …", flush=True)
    try:
        path = kagglehub.dataset_download(DATASET, output_dir=args.download_dir)
    except TypeError:
        path = kagglehub.dataset_download(DATASET)
    root = find_root(path)
    assert root, f"no users/ folder under {path}"
    return root


# ════════════════════════════════════ 2. quaternions & metric ════════════════════════════════════
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw], -1)


def qconj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def small_log(dq):
    """Rotation vector of a small rotation (sign-safe)."""
    return 2 * dq[..., :1] * dq[..., 1:]


def qexp(phi):
    """Quaternion of rotation vector phi, 4th-order polynomial (0.0014° error at 60°)."""
    a = (phi * phi).sum(-1, keepdim=True)
    return torch.cat([1 - a / 8 + a * a / 384, phi * (0.5 - a / 48 + a * a / 3840)], -1)


def angular_velocity(q_prev, q_now):
    return small_log(qmul(qconj(q_prev), q_now)) * FS                 # rad/s, head frame


def loss_angle_deg(q_pred, q_true):
    """Same value as the challenge metric, written with atan2 so the gradient is finite at 0."""
    rel = qmul(qconj(q_pred), q_true)
    v = torch.sqrt((rel[..., 1:] ** 2).sum(-1) + 1e-12)
    return torch.rad2deg(2 * torch.atan2(v, rel[..., 0].abs()))


@torch.no_grad()
def challenge_error(q_pred, q_true):
    """OFFICIAL metric: θ = 2·acos(|q̂·q|) in degrees, both unit-normalised."""
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)
    q_true = q_true / q_true.norm(dim=-1, keepdim=True)
    return torch.rad2deg(2 * torch.acos((q_pred * q_true).sum(-1).abs().clamp(max=1.0)))


# ════════════════════════════════════ 3. models ════════════════════════════════════
@torch.no_grad()
def damped_cv(ctx, beta=1.0, k=1):
    """Baselines: beta=1,k=1 → constant velocity. Returns (B, 20, 4)."""
    q, w = ctx[:, -1], angular_velocity(ctx[:, -1 - k], ctx[:, -1]) / k
    out = []
    for _ in range(HORIZON):
        w = w * beta
        q = qmul(q, qexp(w * DT)); out.append(q)
    return torch.stack(out, 1)


def last_observation(ctx):
    return ctx[:, -1:].expand(-1, HORIZON, -1).clone()


VEL_SCALE = 1 / 3.0        # rad/s → network units (typical head speed 0.5–3 rad/s)
FEATURE_SETS = {"motion": 6, "raw": 4, "relative": 4}
FEATURE_TEXT = {
    "motion": "per-step features that do not depend on the player's heading: head-frame angular velocity (from "
              "consecutive quaternions) and the world 'up' direction seen from the head",
    "raw": "the raw orientation quaternions (w, x, y, z) as recorded, without any transformation",
    "relative": "every orientation expressed relative to the newest one, q_newest⁻¹ ⊗ q_t (sign-canonical, w ≥ 0)",
}


def motion_features(seq):
    """(B, L, 4) → (B, L-1, 6): head-frame angular velocity (scaled) + world 'up' (+Y) seen from the head.
    Both are independent of where the player happens to face (yaw), so the network cannot overfit to it."""
    w = angular_velocity(seq[:, :-1], seq[:, 1:]) * VEL_SCALE
    q = seq[:, 1:]
    up_world = torch.zeros_like(q); up_world[..., 2] = 1.0                  # pure quaternion (0, 0, 1, 0)
    up_head = qmul(qmul(qconj(q), up_world), q)[..., 1:]
    return torch.cat([w, up_head], -1)


def input_features(seq, kind):
    """The encoder input, (B, L, 4) → (B, L-1, F). All variants drop the oldest sample so lengths match.
      motion   : ω (3) + tilt (3)                       — heading-independent (default)
      raw      : the quaternions themselves (4)          — ablation: no feature engineering
      relative : q_newest⁻¹ ⊗ q_t, sign so that w ≥ 0 (4) — heading-independent, keeps the path shape"""
    if kind == "motion":
        return motion_features(seq)
    if kind == "raw":
        return seq[:, 1:]
    if kind == "relative":
        rel = qmul(qconj(seq[:, -1:]).expand_as(seq[:, 1:]), seq[:, 1:])
        return rel * torch.where(rel[..., :1] < 0, -1.0, 1.0)
    raise ValueError(kind)


class Forecaster(nn.Module):
    """Shared input/output for every sequence model.
    in : the last `context` orientations → per-step motion features
    out: 20 cumulative rotation vectors relative to the last orientation,
         = (learned per-horizon gain × last angular velocity) + network correction,
         so training starts from a sensible constant-velocity-like forecast."""
    def __init__(self, context, d, features="motion"):
        super().__init__()
        self.context, self.features, self.n_feat = context, features, FEATURE_SETS[features]
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, HORIZON * 3))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.cv_gain = nn.Parameter(torch.arange(1, HORIZON + 1).float() * DT * 0.7)

    def forward(self, seq):
        seq = seq[:, -self.context:]
        z = self.encode(input_features(seq, self.features))                   # (B, d): summary of the motion
        w_last = angular_velocity(seq[:, -2], seq[:, -1])
        rot = self.cv_gain[None, :, None] * w_last[:, None] + 0.1 * self.head(z).view(-1, HORIZON, 3)
        q_last = seq[:, -1:].expand(-1, HORIZON, -1)
        pred = qmul(q_last, qexp(rot))
        return pred / pred.norm(dim=-1, keepdim=True)                         # (B, 20, 4)


class AttentionBlock(nn.Module):
    """Pre-norm transformer block with hand-written multi-head attention (MatMul + Softmax only, so it exports
    cleanly to ONNX and never hits PyTorch's fused fast-path kernels)."""
    def __init__(self, d, heads, ff_mult=2, dropout=0.0):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                                   # (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = torch.softmax((q @ k.transpose(-1, -2)) * (D // self.h) ** -0.5, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.n2(x)))


class TransformerModel(Forecaster):
    """Transformer encoder with a motion-aware input embedding:
      1. richer per-step features, all heading-independent: angular velocity ω (3), head tilt (3),
         speed |ω| (1) and angular acceleration Δω (3)  →  10 values per 10 ms step;
      2. PATCH embedding: a strided Conv1d turns every `patch` steps (default 5 = 50 ms) into one token,
         so a 1 s context becomes 20 tokens (short sequence → cheap attention, smoother inputs);
      3. learned positional embedding = how long ago each patch happened (recency);
      4. a READOUT token appended after the patches, initialised from the newest step's features, attends to
         the whole history; its output feeds the shared forecast head."""
    NAME = "Transformer"

    def __init__(self, context, d=64, heads=4, layers=3, patch=5, dropout=0.0, features="motion"):
        super().__init__(context, d, features)
        self.patch = patch
        self.n_tok = (context - 1) // patch
        self.N_IN = self.n_feat + (4 if features == "motion" else 0)      # + speed, angular acceleration
        self.feat_norm = nn.LayerNorm(self.N_IN)
        self.embed = nn.Conv1d(self.N_IN, d, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, d) * 0.02)
        self.readout = nn.Linear(self.N_IN, d)
        self.readout_bias = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.Sequential(*[AttentionBlock(d, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def encode(self, x):                                                    # x: (B, L, F) input features
        if self.features == "motion":
            w = x[..., :3]
            dw = torch.cat([torch.zeros_like(w[:, :1]), w[:, 1:] - w[:, :-1]], 1) * 10.0     # angular acceleration
            speed = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-8)
            x = torch.cat([x, speed, dw], -1)                                   # (B, L, 10)
        f = self.feat_norm(x)
        f_patch = f[:, f.shape[1] - self.n_tok * self.patch:]                    # newest full patches
        tok = self.embed(f_patch.transpose(1, 2)).transpose(1, 2) + self.pos    # (B, n_tok, d)
        ro = self.readout(f[:, -1:]) + self.readout_bias                        # (B, 1, d)
        z = self.blocks(torch.cat([tok, ro], 1))
        return self.norm(z[:, -1])


MODELS = {"transformer": TransformerModel}


def count_macs(model, context):
    """MACs of ONE prediction (one call on the last `context` samples), counted with forward hooks."""
    total = [0]
    def lin(m, i, o): total[0] += m.in_features * m.out_features * (o.numel() // o.shape[-1])
    def conv(m, i, o): total[0] += (m.in_channels // m.groups) * m.out_channels * m.kernel_size[0] * o.shape[-1]
    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear): hooks.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.Conv1d): hooks.append(mod.register_forward_hook(conv))
    with torch.no_grad():
        model(torch.tensor([[[1.0, 0, 0, 0]]]).expand(1, context, 4).clone())
    for h in hooks: h.remove()
    if isinstance(model, TransformerModel):                                     # attention score / mix matmuls
        T, D = model.n_tok + 1, model.norm.normalized_shape[0]
        total[0] += len(model.blocks) * 2 * T * T * D
    feats = (context - 1) * (16 * 3)                                            # quaternion maths for the features
    return total[0] + feats + HORIZON * 30


# ════════════════════════════════════ 4. evaluation ════════════════════════════════════
@torch.no_grad()
def predict_windows(fn, windows, context, device, batch=4096):
    out = []
    for i in range(0, len(windows), batch):
        ctx = windows[i:i + batch, CONTEXT - context:CONTEXT].to(device)
        out.append(fn(ctx).cpu())
    return torch.cat(out)


def user_bootstrap(err_a, err_b, users, n=1000, seed=0):
    """95 % CI of mean(b) − mean(a), resampling whole users."""
    d = pd.Series((err_b - err_a).numpy()).groupby(users).agg(["sum", "count"])
    rng, diffs = np.random.default_rng(seed), []
    for _ in range(n):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        diffs.append(s["sum"].sum() / s["count"].sum())
    return np.percentile(diffs, [2.5, 97.5])



# ════════════════════════════════════ 5. export & package ════════════════════════════════════
def export_onnx(module, inputs, path, input_names, output_names):
    kw = dict(input_names=input_names, output_names=output_names, opset_version=17)
    try:
        torch.onnx.export(module, inputs, str(path), dynamo=False, **kw)      # classic exporter
    except Exception as e:
        print(f"  classic ONNX exporter unavailable ({type(e).__name__}); using dynamo exporter", flush=True)
        torch.onnx.export(module, inputs, str(path), dynamo=True, **kw)


RUN_INFERENCE = '''\
"""run_inference.py — {team} head-pose prediction (9th ABC Challenge).

    python run_inference.py --model_dir . --input sample.npy --output pred.npy

Input  context: float32 [400, 4] unit quaternions, scalar-first (w, x, y, z), oldest first.
Output prediction: float32 [20, 4] unit quaternions (w, x, y, z) for the next 20 samples (10 ms apart).
The model is stateless: each call reads the last {k} context samples. No network access.
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_model(model_dir: str) -> object:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(Path(model_dir) / "model_fp32.onnx"), opts,
                                providers=["CPUExecutionProvider"])


def predict(model, context: np.ndarray) -> np.ndarray:
    ctx = np.asarray(context, dtype=np.float32)
    if ctx.shape != (400, 4):
        raise ValueError(f"context must be [400, 4], got {{ctx.shape}}")
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(ctx, axis=1, keepdims=True)
    ctx = np.where(norm > 1e-6, ctx / np.maximum(norm, 1e-6), np.array([1, 0, 0, 0], np.float32))
    pred = model.run(None, {{"context": ctx[None]}})[0][0]
    return pred.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=".")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    np.save(a.output, predict(load_model(a.model_dir), np.load(a.input)))
'''


DESCRIPTIONS = {
    "Transformer": "transformer encoder (pre-norm blocks, multi-head attention) over 50 ms patch tokens of motion "
                   "features (angular velocity, tilt, speed, angular acceleration) with learned recency embeddings "
                   "and a readout token initialised from the newest sample.",
}


def write_package(args, out, model, k, metrics, eff, ops, cfg):
    name = model.NAME
    pkg = out / f"{args.team}_headpose"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    shutil.copy(out / "model_fp32.onnx", pkg / "model_fp32.onnx")
    (pkg / "run_inference.py").write_text(RUN_INFERENCE.format(team=args.team, k=k))
    import onnxruntime
    (pkg / "requirements.txt").write_text(f"numpy=={np.__version__}\nonnxruntime=={onnxruntime.__version__}\n")
    res = metrics["results"]["test"]
    main = res[cfg["main_name"]]
    rows = "\n".join(f"        | {n} | {res[n]['mean_200ms']:.3f} | {res[n]['median_200ms']:.3f} | {res[n]['p95_200ms']:.3f} |"
                     for n in ["constant velocity", "last observation", "damped CV"])
    (pkg / "model_card.md").write_text(textwrap.dedent(f"""\
        # Model card — {args.team}: {name} head-pose predictor

        **Task.** 400 past head orientations (unit quaternions, w,x,y,z, 100 Hz) → next 20 orientations (200 ms).

        ## Architecture
        Inputs: the last {k} orientations; the encoder sees {FEATURE_TEXT[cfg.get('features', 'motion')]}.
        Encoder: {DESCRIPTIONS[name]}
        Output: 20 rotation vectors relative to the last orientation = learned per-horizon multiple of the last
        angular velocity + network correction, composed onto the last orientation and unit-normalised.

        | item | value |
        |---|---|
        | trainable parameters | {eff['parameters']:,} |
        | MACs per prediction (one call, {k}-sample context) | {eff['macs_per_call']:,} |
        | context length used | {k} samples (the last {k} of the 400 provided) |
        | state tensors | none (stateless: every call reads the context window) |
        | quantisation / pruning / distillation | none (FP32) |

        File: `model_fp32.onnx` — input `context` [1, 400, 4] → output `prediction` [1, 20, 4].
        ONNX operators: {', '.join(ops)}.

        ## Training data and preprocessing
        BOXRR-23 (Nair et al., IEEE TVCG 2024), Beat Saber head tracking, filtered and cleaned by the team's
        preprocessing (challenge filters; unit quaternions; hemisphere-aligned; 100 Hz grid; gaps split, not
        interpolated). Users split 80/10/10 (train/val/test) by a hash of the user ID; no user appears in two splits.
        {metrics['data']['train_users']:,} training users ({metrics['data']['train_hours']:.0f} h, random crops),
        {metrics['data']['val_users']:,} validation users (checkpoint selection), {metrics['data']['test_users']:,}
        test users (reported once). Loss: mean geodesic error at the 20th predicted sample
        (+{cfg['aux']} × mean over all horizons). AdamW, one-cycle LR {cfg['lr']}, batch {cfg['batch']}, {cfg['steps']} steps.

        ## Test results (held-out test users, {metrics['data']['test_windows']:,} windows)
        | model | mean error @20th sample (°) | median (°) | p95 (°) |
        |---|---|---|---|
{rows}
        | **{name} (this model)** | **{main['mean_200ms']:.3f}** | {main['median_200ms']:.3f} | {main['p95_200ms']:.3f} |

        CPU latency (onnxruntime, 1 thread): {eff['onnx_ms_context_call']:.3f} ms per call.

        ## Known failure modes
        * Largest errors at motion onsets and abrupt reversals (see by_regime_test.csv): nothing in the past
          {k / FS:.1f} s announces a movement that has not started.
        * Trained and validated on Beat Saber only; other applications may move differently.
        * Assumes clean, uniformly sampled 100 Hz input; gaps or duplicated samples appear as sudden velocity changes.
        """))
    (pkg / "README.md").write_text(textwrap.dedent(f"""\
        # {args.team} — head-pose prediction (9th ABC Challenge)

        ```bash
        pip install -r requirements.txt
        python run_inference.py --model_dir . --input sample.npy --output pred.npy
        ```

        ```python
        from run_inference import load_model, predict
        model = load_model(".")
        prediction = predict(model, context)      # context float32 [400, 4] (w,x,y,z) → float32 [20, 4]
        ```

        `model_fp32.onnx` is the submitted {name} model (stateless). Details and metrics: `model_card.md`.
        """))
    np.save(pkg / "sample.npy", metrics.pop("_sample_context"))
    return pkg


def zip_package(out, pkg):
    zpath = out / f"{pkg.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            if p.name != "__pycache__" and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(out))
    return zpath


def resolve_device(name):
    """'auto' → first GPU if available, else CPU; otherwise validate 'cpu', 'cuda', 'cuda:N'."""
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit(f"--device {name}: CUDA is not available on this machine")
        idx = 0 if dev.index is None else dev.index
        if idx >= torch.cuda.device_count():
            sys.exit(f"--device {name}: only {torch.cuda.device_count()} GPU(s) visible (cuda:0 … cuda:{torch.cuda.device_count() - 1})")
        torch.cuda.set_device(idx)                     # so any implicit 'cuda' tensor lands on the same GPU
        return f"cuda:{idx}"
    return str(dev)


# ════════════════════════════════════ main ════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-model", type=int, default=64, help="token width")
    ap.add_argument("--blocks", type=int, default=3, help="transformer blocks")
    ap.add_argument("--heads", type=int, default=4, help="attention heads (must divide --d-model)")
    ap.add_argument("--patch", type=int, default=5, help="samples per token (5 = 50 ms)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--team", default="TEAMNAME", help="registered team name (used for the ZIP)")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="motion",
                    help="encoder input: motion (ω + tilt, default) | raw (quaternions) | relative (q_newest⁻¹ ⊗ q_t)")
    ap.add_argument("--data", help="folder containing users/ (skips detection / download)")
    ap.add_argument("--download-dir", default="data/headvane", help="where kagglehub downloads the corpus")
    ap.add_argument("--work", default="work", help="cache folder (fast local disk)")
    ap.add_argument("--out", default="runs/transformer", help="output folder")
    ap.add_argument("--context", type=int, default=100, help="samples the model reads (≤ 400)")
    ap.add_argument("--span", type=int, default=100, help="random-crop range inside each training chunk")
    ap.add_argument("--chunks-per-user", type=int, default=300)
    ap.add_argument("--val-windows-per-user", type=int, default=40, help="also used for test users")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of all-horizon error (0 = pure challenge metric)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: stop after this many evaluations without improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, default=1e-3, help="improvement (°) that counts for early stopping")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:0 | cuda:1 | … (auto = first GPU if available)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the pipeline")
    args = ap.parse_args()

    device = resolve_device(args.device)
    args.model = "transformer"
    assert args.d_model % args.heads == 0, "--heads must divide --d-model"
    model_kw = dict(d=args.d_model, layers=args.blocks, heads=args.heads, patch=args.patch, dropout=args.dropout,
                    features=args.features)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    k = args.context
    assert 8 <= k <= CONTEXT
    assert (k - 1) // args.patch >= 2, "context too short for this patch size"
    train_len = k + args.span + HORIZON                      # each chunk gives span+1 possible windows (random crop)

    # ── data (same cache format and user-level split as the other trainers) ──
    root = connect(args)
    files = sorted(root.rglob("*.h5"))
    if args.smoke:
        files = random.Random(args.seed).sample(files, min(150, len(files)))
        args.chunks_per_user, args.val_windows_per_user = 50, 10
        args.steps, args.eval_every, args.log_every, args.batch = 30, 10, 5, 32
        args.patience = min(args.patience, 1) if args.patience else 0
    print(f"model {args.model} | device {device} | corpus {root} | {len(files):,} files", flush=True)
    caps = {"train": args.chunks_per_user, "val": args.val_windows_per_user, "test": args.val_windows_per_user}
    tag = f"{'smoke' if args.smoke else 'full'}_L{train_len}_c{args.chunks_per_user}_v{args.val_windows_per_user}_s{args.seed}"
    cache = work / f"cache_{tag}"
    meta = build_cache(files, cache, train_len, caps, args.workers, args.seed)
    for s in SPLITS:
        assert meta[f"n_{s}"], f"empty {s} split — use more files"
    TRAIN = np.memmap(cache / "train.bin", dtype=np.float32, mode="r", shape=(meta["n_train"], train_len, 4))
    load = lambda s: torch.from_numpy(np.array(np.memmap(cache / f"{s}.bin", dtype=np.float32, mode="r",
                                                         shape=(meta[f"n_{s}"], CONTEXT + HORIZON, 4))))
    WIN = {"val": load("val"), "test": load("test")}
    USERS = {s: np.array(meta[f"owner_{s}"]) for s in ("val", "test")}
    data_info = {"files": len(files), "train_chunks": meta["n_train"],
                 "train_hours": meta["n_train"] * train_len / FS / 3600,
                 **{f"{s}_users": meta[f"{s}_users"] for s in SPLITS},
                 **{f"{s}_windows": meta[f"n_{s}"] for s in ("val", "test")}}
    print(" | ".join(f"{s}: {meta[f'{s}_users']:,} users" for s in SPLITS) +
          f" | train {data_info['train_hours']:.0f} h | val {meta['n_val']:,} / test {meta['n_test']:,} windows", flush=True)

    rng = np.random.default_rng(args.seed)
    offs = torch.arange(k + HORIZON)
    def train_batch():
        idx = np.sort(rng.choice(len(TRAIN), args.batch, replace=False))
        chunk = torch.from_numpy(np.array(TRAIN[idx]))                          # (B, train_len, 4)
        start = torch.from_numpy(rng.integers(0, args.span + 1, args.batch))    # random crop per sample
        return torch.gather(chunk, 1, (start[:, None] + offs)[..., None].expand(-1, -1, 4))   # (B, k+20, 4)
    val_small = WIN["val"][np.random.default_rng(1).choice(len(WIN["val"]), min(3000, len(WIN["val"])), replace=False)]

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(out / "tensorboard")
    tb.add_text("data", "\n".join(f"* **{k_}**: {v}" for k_, v in data_info.items()))
    tb.add_text("config", "```\n" + json.dumps(vars(args), indent=2) + "\n```")

    # ── baselines (damped CV tuned on training users) ──
    tune = torch.from_numpy(np.array(TRAIN[np.sort(rng.choice(len(TRAIN), min(5000, len(TRAIN)), replace=False))]))
    t_ctx, t_true = tune[:, :train_len - HORIZON], tune[:, -1]
    grid = [(b, kk) for b in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.0) for kk in (1, 2, 3, 5)]
    beta, kv = min(grid, key=lambda g: challenge_error(damped_cv(t_ctx, *g)[:, -1], t_true).mean().item())
    print(f"damped CV tuned on train users: beta={beta}, k={kv}", flush=True)
    baseline_fns = {"last observation": last_observation,
                    "constant velocity": lambda c: damped_cv(c, 1.0, 1),
                    "damped CV": lambda c: damped_cv(c, beta, kv)}
    base_val_small = {n: challenge_error(predict_windows(fn, val_small, CONTEXT, device), val_small[:, CONTEXT:])[:, -1].mean().item()
                      for n, fn in baseline_fns.items()}

    # ── train ──
    net = MODELS[args.model](k, **model_kw).to(device)
    NAME = net.NAME
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    ckpt, best, hist, t0 = out / f"{args.model}_best.pt", float("inf"), [], time.time()
    bad_evals, stopped_at = 0, None

    def net_errors(windows):
        net.eval()
        e = challenge_error(predict_windows(net, windows, k, device), windows[:, CONTEXT:])
        net.train()
        return e

    print(f"{NAME}: {sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
    for step in range(1, args.steps + 1):
        w = train_batch().to(device, non_blocking=True)
        err = loss_angle_deg(net(w[:, :k]), w[:, k:])
        loss = err[:, -1].mean() + args.aux * err.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == 1:
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/err_20th_sample_deg", err[:, -1].mean().item(), step)
            tb.add_scalar("train/err_10th_sample_deg", err[:, 9].mean().item(), step)
            tb.add_scalar("train/err_all_horizons_deg", err.mean().item(), step)
            tb.add_scalar("train/learning_rate", sched.get_last_lr()[0], step)
            tb.add_scalar("train/grad_norm", float(grad_norm), step)
            tb.add_scalar("train/steps_per_second", step / (time.time() - t0), step)
        if step % args.eval_every == 0 or step == args.steps:
            e = net_errors(val_small)
            v = e[:, -1].mean().item()
            if v < best - args.min_delta:
                best, bad_evals = v, 0
                torch.save(net.state_dict(), ckpt)
            else:
                bad_evals += 1
                if v < best:                                    # tiny gain: keep the weights, still counts as "no progress"
                    best = v; torch.save(net.state_dict(), ckpt)
            tb.add_scalar("val/err_20th_sample_deg", v, step)
            tb.add_scalar("val/err_10th_sample_deg", e[:, 9].mean().item(), step)
            tb.add_scalar("val/best_err_20th_sample_deg", best, step)
            tb.add_scalars("val/compare_err_20th_sample_deg", {NAME: v, **base_val_small}, step)
            hist.append({"step": step, "train_loss": loss.item(), "val_err20": v, "seconds": time.time() - t0})
            print(f"step {step:5d}/{args.steps} | train {loss.item():.3f}° | val@20 {v:.3f}° | best {best:.3f}° "
                  f"| {time.time() - t0:.0f}s", flush=True)
            tb.add_scalar("val/evals_without_improvement", bad_evals, step)
            tb.flush()
            if args.patience and bad_evals >= args.patience:
                stopped_at = step
                print(f"early stopping at step {step}: no improvement > {args.min_delta}° for {args.patience} "
                      f"evaluations ({args.patience * args.eval_every} steps); best val@20 {best:.3f}°", flush=True)
                break
    net.load_state_dict(torch.load(ckpt, map_location=device)); net.eval()
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    # ── benchmark: validation and TEST (used once) ──
    print("benchmarking on all validation and test windows …", flush=True)
    main_name = f"{NAME} (context {k})"
    errors, preds_test, results = {}, {}, {}
    for split in ("val", "test"):
        W = WIN[split]
        preds = {n: predict_windows(fn, W, CONTEXT, device) for n, fn in baseline_fns.items()}
        with torch.no_grad():
            preds[main_name] = predict_windows(net, W, k, device)
        errors[split] = {n: challenge_error(p, W[:, CONTEXT:]) for n, p in preds.items()}
        if split == "test":
            preds_test = preds
        res = {n: {"mean_200ms": e[:, -1].mean().item(), "mean_100ms": e[:, 9].mean().item(),
                   "median_200ms": e[:, -1].median().item(), "p95_200ms": e[:, -1].quantile(0.95).item(),
                   "mean_all_horizons": e.mean().item()} for n, e in errors[split].items()}
        ref = res["constant velocity"]["mean_200ms"]
        for r in res.values():
            r["vs_constant_velocity_pct"] = (r["mean_200ms"] / ref - 1) * 100
        results[split] = res
        pd.DataFrame(res).T.sort_values("mean_200ms").to_csv(out / f"results_{split}.csv")
        pd.DataFrame({n: e.mean(0).numpy() for n, e in errors[split].items()},
                     index=pd.Index(np.arange(1, HORIZON + 1) * 10, name="horizon_ms")).to_csv(out / f"per_horizon_{split}.csv")

    compare = ["last observation", "constant velocity", "damped CV", main_name]
    E = errors["test"]
    speed = torch.rad2deg(angular_velocity(WIN["test"][:, CONTEXT - 2], WIN["test"][:, CONTEXT - 1]).norm(dim=-1)).numpy()
    bands = pd.cut(speed, [0, 5, 50, 150, np.inf], labels=["still <5°/s", "slow 5–50", "moving 50–150", "fast >150"], right=False)
    by_regime = pd.DataFrame({n: E[n][:, -1].numpy() for n in compare}).groupby(bands, observed=False).mean()
    by_regime.insert(0, "share_%", pd.Series(bands).value_counts(normalize=True, sort=False).values * 100)
    by_regime.to_csv(out / "by_regime_test.csv")

    boot = []
    for other in ["last observation", "constant velocity", "damped CV"]:
        lo, hi = user_bootstrap(E[main_name][:, -1], E[other][:, -1], USERS["test"])
        boot.append({"comparison": f"{other} − {NAME} (test)", "ci95_low_deg": lo, "ci95_high_deg": hi,
                     "significant": bool(lo > 0)})
    pd.DataFrame(boot).to_csv(out / "bootstrap_test.csv", index=False)

    for split in ("val", "test"):
        for n, r in results[split].items():
            tb.add_scalar(f"final_{split}/{n.replace(' ', '_')}_err_20th_sample_deg", r["mean_200ms"], args.steps)
        for h in range(HORIZON):
            tb.add_scalars(f"final_{split}/error_vs_horizon_deg",
                           {n.replace(" ", "_"): errors[split][n][:, h].mean().item() for n in compare}, (h + 1) * 10)
        table = pd.DataFrame(results[split]).T.round(3)
        try:
            tb.add_text(f"final_{split}/results", table.to_markdown())
        except ImportError:
            tb.add_text(f"final_{split}/results", "```\n" + table.to_string() + "\n```")
    tb.add_text("final_test/by_regime", "```\n" + by_regime.round(3).to_string() + "\n```")
    tb.add_text("final_test/bootstrap", "\n".join(f"* {b['comparison']}: [{b['ci95_low_deg']:+.3f}, "
                                                   f"{b['ci95_high_deg']:+.3f}]°" for b in boot))

    # ── export ONNX + check ──
    print("exporting ONNX …", flush=True)
    import onnx, onnxruntime as ort
    TEST = WIN["test"]
    cpu_net = MODELS[args.model](k, **model_kw); cpu_net.load_state_dict(net.state_dict()); cpu_net.eval()
    if hasattr(cpu_net, "freeze"):
        cpu_net.freeze()
    with torch.no_grad():
        export_onnx(cpu_net, (TEST[:1, :CONTEXT].clone(),), out / "model_fp32.onnx", ["context"], ["prediction"])
    ops = sorted({nd.op_type for nd in onnx.load(out / "model_fp32.onnx").graph.node})
    safe = {"Conv", "MatMul", "Gemm", "LayerNormalization", "BatchNormalization", "Relu", "Gelu", "Add", "Mul",
            "Concat", "Slice", "Reshape"}
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s_ctx = ort.InferenceSession(str(out / "model_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    n_chk = min(200, len(TEST))
    onnx_pred = np.concatenate([s_ctx.run(None, {"context": TEST[i:i + 1, :CONTEXT].numpy()})[0] for i in range(n_chk)])
    torch_pred = preds_test[main_name][:n_chk].numpy()
    checks = {"max_abs_diff_onnx_vs_torch": float(np.abs(onnx_pred - torch_pred).max()),
              "onnx_mean_err20_first_test_windows": challenge_error(torch.from_numpy(onnx_pred), TEST[:n_chk, CONTEXT:])[:, -1].mean().item(),
              "torch_mean_err20_first_test_windows": E[main_name][:n_chk, -1].mean().item(),
              "nan_or_inf": bool(not np.isfinite(onnx_pred).all()), "output_shape": list(onnx_pred.shape[1:])}

    def timeit(fn, n=300):
        for _ in range(20): fn()
        ts = []
        for _ in range(n):
            s0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - s0)
        return float(np.median(ts) * 1e3)

    x_ctx = TEST[:1, :CONTEXT].numpy()
    eff = {"parameters": sum(p.numel() for p in cpu_net.parameters() if p.requires_grad),
           "macs_per_call": count_macs(cpu_net, k), "context_samples": k,
           "onnx_ms_context_call": timeit(lambda: s_ctx.run(None, {"context": x_ctx})),
           "onnx_size_kb": (out / "model_fp32.onnx").stat().st_size / 1024,
           "onnx_ops": ops, "onnx_ops_outside_planning_doc_safe_list": [o for o in ops if o not in safe]}
    eff["macs_context_call"] = eff["macs_per_call"]                 # same key as the other trainers
    pd.Series({k_: v for k_, v in eff.items() if not isinstance(v, list)}).to_csv(out / "efficiency.csv")
    for k_ in ("parameters", "macs_per_call", "onnx_ms_context_call"):
        tb.add_scalar(f"efficiency/{k_}", eff[k_], args.steps)

    # ── metrics.json, config, package ──
    cfg = {**vars(args), "model_kwargs": model_kw, "stopped_at_step": stopped_at, "train_len": train_len, "main_name": main_name, "damped_cv": {"beta": beta, "k": kv},
           "device": device, "torch": torch.__version__, "context": k}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics = {"model": NAME, "data": data_info, "results": results, "bootstrap_test": boot,
               "by_regime_test": by_regime.round(4).to_dict(), "efficiency": eff, "onnx_checks": checks,
               "best_val_err20_small": best, "_sample_context": TEST[0, :CONTEXT].numpy()}
    pkg = write_package(args, out, cpu_net, k, metrics, eff, ops, cfg)
    r = subprocess.run([sys.executable, "run_inference.py", "--model_dir", ".", "--input", "sample.npy",
                        "--output", "pred_test.npy"], cwd=pkg, capture_output=True, text=True)
    ok = r.returncode == 0 and (pkg / "pred_test.npy").exists()
    if ok:
        p = np.load(pkg / "pred_test.npy")
        ok = p.shape == (20, 4) and p.dtype == np.float32 and bool(np.isfinite(p).all())
        (pkg / "pred_test.npy").unlink()
    if not ok:
        print("run_inference.py test FAILED:\n", r.stdout, r.stderr)
    (pkg / "sample.npy").unlink(missing_ok=True)
    zpath = zip_package(out, pkg)
    checks["package_run_inference_ok"] = bool(ok)
    checks["zip_size_mb"] = round(zpath.stat().st_size / 1024 ** 2, 2)
    assert checks["zip_size_mb"] < 500, "ZIP exceeds the 500 MB limit"
    metrics.pop("_sample_context", None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ── figures, hparams, report ──
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 4))
        h = pd.DataFrame(hist); ax[0].plot(h.step, h.val_err20, marker="o", label=f"{NAME} (val)")
        for n, ls in (("damped CV", ":"), ("last observation", "--")):
            ax[0].axhline(base_val_small[n], color="gray", ls=ls, label=n)
        ax[0].set(xlabel="step", ylabel="val error @20th (°)", title="Training"); ax[0].legend()
        for n in compare:
            ax[1].plot(np.arange(1, HORIZON + 1) * 10, E[n].mean(0), marker=".", label=n)
        ax[1].set(xlabel="horizon (ms)", ylabel="mean error (°)", title="Test: error vs horizon"); ax[1].legend(fontsize=8)
        by_regime.drop(columns="share_%").plot.bar(ax=ax[2], rot=0)
        ax[2].set(ylabel="mean error @200 ms (°)", title="Test: by motion regime"); ax[2].legend(fontsize=7)
        plt.tight_layout(); fig.savefig(out / "fig_benchmark.png", dpi=130)
        tb.add_figure("final_test/benchmark", fig, args.steps)
    except Exception as e:
        print("figures skipped:", e)
    hp = {k_: v for k_, v in vars(args).items() if isinstance(v, (int, float, str, bool)) and k_ not in ("team",)}
    tb.add_hparams(hp, {"hparam/val_err20": results["val"][main_name]["mean_200ms"],
                        "hparam/test_err20": results["test"][main_name]["mean_200ms"],
                        "hparam/parameters": eff["parameters"]}, run_name="hparams")
    tb.close()

    lines = [f"# {NAME} benchmark — {args.team}", "",
             f"Train {data_info['train_users']:,} users ({data_info['train_hours']:.0f} h) · validation "
             f"{data_info['val_users']:,} users · test {data_info['test_users']:,} users ({data_info['test_windows']:,} "
             "windows, reported once). Splits are user-disjoint. Metric: mean 2·acos(|q̂·q|) at the 20th sample.", ""]
    for split in ("test", "val"):
        t = pd.DataFrame(results[split]).T.sort_values("mean_200ms")
        lines += [f"## {split.upper()}", "", "| model | mean @200 ms (°) | mean @100 ms (°) | median (°) | p95 (°) | vs CV |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {n} | {r.mean_200ms:.3f} | {r.mean_100ms:.3f} | {r.median_200ms:.3f} | {r.p95_200ms:.3f} | "
                  f"{r.vs_constant_velocity_pct:+.1f} % |" for n, r in t.iterrows()]
        lines += [""]
    lines += ["Per-user bootstrap on TEST (95 % CI of baseline − model; > 0 means the model is better):", ""]
    lines += [f"* {b['comparison']}: [{b['ci95_low_deg']:+.3f}, {b['ci95_high_deg']:+.3f}]° "
              f"{'✓ significant' if b['significant'] else '· not significant'}" for b in boot]
    lines += ["", f"Parameters {eff['parameters']:,} · MACs/call {eff['macs_per_call']:,} · "
                  f"ONNX {eff['onnx_ms_context_call']:.3f} ms/call ({k}-sample context, CPU, 1 thread)",
              f"Operators outside the planning doc's safe list: {eff['onnx_ops_outside_planning_doc_safe_list']}",
              f"ONNX checks: {checks}", "", f"Package: `{zpath}`", "",
              f"TensorBoard: `tensorboard --logdir {out / 'tensorboard'}`", "", "![](fig_benchmark.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    t = pd.DataFrame(results["test"]).T.sort_values("mean_200ms")
    print("\nTEST (held-out users):")
    print(t[["mean_200ms", "mean_100ms", "median_200ms", "p95_200ms", "vs_constant_velocity_pct"]].round(3).to_string())
    print("\nbootstrap:", *[f"{b['comparison']}: [{b['ci95_low_deg']:+.3f}, {b['ci95_high_deg']:+.3f}]" for b in boot], sep="\n  ")
    print(f"\n{NAME}: params {eff['parameters']:,} | ONNX {eff['onnx_ms_context_call']:.3f} ms/call | "
          f"package test {'OK' if checks['package_run_inference_ok'] else 'FAILED'}")
    print(f"everything saved in {out.resolve()}  |  submission: {zpath}")


if __name__ == "__main__":
    main()