#!/usr/bin/env python
"""
train_kalmannet.py — 9th ABC Challenge: train, benchmark and package KalmanNet head-pose prediction.

    python train_kalmannet.py --team MYTEAM                        # full run
    python train_kalmannet.py --team MYTEAM --smoke                # 5-minute end-to-end check first
    python train_kalmannet.py --team MYTEAM --data /path/to/corpus # data already on disk

What it does
  1. Data      finds the preprocessed corpus (--data, an attached Kaggle dataset, or downloads it
               with kagglehub), then scans ALL files once and caches, for EVERY user, up to
               --chunks-per-user training chunks (train users) or --val-windows-per-user
               400+20 windows (val and test users). Splits are USER-disjoint: 80/10/10 by a hash of the
               user ID (val = the EDA's validation users); no user appears in two splits.
  2. Train     KalmanNet with the challenge metric as loss (mean geodesic error at the 20th sample);
               checkpoint chosen on VALIDATION; everything logged to TensorBoard.
  3. Benchmark on all val and all TEST users (test is used once, at the end): last observation,
               constant velocity (required baseline), damped constant velocity (tuned on train users)
               and KalmanNet at several context lengths; per-horizon curves, motion regimes,
               per-user bootstrap intervals.
  4. Efficiency trainable parameters, MACs per streaming step / per prediction, CPU latency.
  5. Package   ONNX models + run_inference.py (load_model / predict) + requirements.txt +
               model_card.md + README.md, tested in a subprocess, zipped as TEAMNAME_headpose.zip.

Everything lands in --out (default runs/kalmannet):
  tensorboard/  metrics.json  results_{val,test}.csv  per_horizon_{val,test}.csv  by_regime_test.csv
  bootstrap_test.csv  efficiency.csv  history.csv  config.json  report.md
  fig_benchmark.png  kalmannet_best.pt  TEAMNAME_headpose/  TEAMNAME_headpose.zip
  View the logs with:  tensorboard --logdir runs/kalmannet/tensorboard
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


class GRUCell(nn.Module):
    """GRU written with Linear + sigmoid/tanh, so the ONNX graph contains no GRU operator."""
    def __init__(self, n_in, n_h):
        super().__init__()
        self.x2h, self.h2h = nn.Linear(n_in, 3 * n_h), nn.Linear(n_h, 3 * n_h)

    def forward(self, x, h):
        xr, xz, xn = self.x2h(x).chunk(3, -1); hr, hz, hn = self.h2h(h).chunk(3, -1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        return (1 - z) * torch.tanh(xn + r * hn) + z * h


def unit(v):
    return v * torch.rsqrt((v * v).sum(-1, keepdim=True) + 1e-8)


class KalmanNet(nn.Module):
    """KalmanNet (Revach et al., IEEE TSP 2022; architecture as in SongJgit/KalmanNet4SensorFusion)
    on a rotation-motion filter:
        observation y_t = head-frame angular velocity from consecutive orientations (3)
        state       x_t = [ω_t, d_t]  angular velocity and its change per sample (6)
        motion      ω ← ω + d,  d ← a·d   (learned decay a)
        prediction  roll the motion model 20 steps, composing rotations onto the last orientation
    The GRUs learn the 6×3 Kalman gain. State tensors are zero at the start of a session."""
    m, n = 6, 3

    def __init__(self, in_mult=5, out_mult=40):
        super().__init__()
        m, n = self.m, self.n
        self.layout = {"q_prev": 4, "x": m, "x_prev": m, "prior_prev": m, "y_prev": n,
                       "h_Q": m * m, "h_Sigma": m * m, "h_S": n * n}
        self.state_size = sum(self.layout.values())
        self.decay_logit = nn.Parameter(torch.full((3,), 2.0))
        self.FC5 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU()); self.GRU_Q = GRUCell(m * in_mult, m * m)
        self.FC6 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU()); self.GRU_Sigma = GRUCell(m * m + m * in_mult, m * m)
        self.FC1 = nn.Sequential(nn.Linear(m * m, n * n), nn.ReLU())
        self.FC7 = nn.Sequential(nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU())
        self.GRU_S = GRUCell(n * n + 2 * n * in_mult, n * n)
        d2 = n * n + m * m
        self.FC2 = nn.Sequential(nn.Linear(d2, d2 * out_mult), nn.ReLU(), nn.Linear(d2 * out_mult, m * n))
        self.FC3 = nn.Sequential(nn.Linear(n * n + m * n, m * m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(2 * m * m, m * m), nn.ReLU())
        nn.init.zeros_(self.FC2[2].weight); nn.init.zeros_(self.FC2[2].bias)
        self.base_gain = nn.Parameter(torch.zeros(m, n))     # gain 0 at start ≡ last-observation predictor

    def decay(self):
        return torch.sigmoid(self.decay_logit)

    def motion_matrix(self):
        a = self.decay()
        I, Z = torch.eye(3, device=a.device), torch.zeros(3, 3, device=a.device)
        return torch.cat([torch.cat([I, I], 1), torch.cat([Z, I * a], 1)], 0)

    def init_state(self, b, device="cpu"):
        return torch.zeros(b, self.state_size, device=device)

    def step(self, q_now, state, F):
        s, i = {}, 0
        for k, size in self.layout.items():
            s[k] = state[:, i:i + size]; i += size
        valid = ((s["q_prev"] ** 2).sum(-1, keepdim=True) > 0.5).float()      # first sample: no velocity
        y = angular_velocity(s["q_prev"], q_now) * valid
        prior = s["x"] @ F.T
        innov = (y - prior[:, :3]) * valid
        f1, f2 = unit((y - s["y_prev"]) * valid), unit(innov)
        f3, f4 = unit(s["x"] - s["x_prev"]), unit(s["x"] - s["prior_prev"])
        h_Q = self.GRU_Q(self.FC5(f3), s["h_Q"])
        h_Sigma = self.GRU_Sigma(torch.cat([h_Q, self.FC6(f4)], -1), s["h_Sigma"])
        h_S = self.GRU_S(torch.cat([self.FC1(h_Sigma), self.FC7(torch.cat([f1, f2], -1))], -1), s["h_S"])
        g = self.FC2(torch.cat([h_Sigma, h_S], -1))
        h_Sigma = self.FC4(torch.cat([h_Sigma, self.FC3(torch.cat([h_S, g], -1))], -1))
        K = (g + self.base_gain.flatten()).view(-1, self.m, self.n)
        x = prior + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
        new = torch.cat([q_now, x, s["x"], prior, y, h_Q, h_Sigma, h_S], 1)
        return x, new

    def predict(self, q_now, x):
        a = self.decay()
        w, d = x[..., :3], x[..., 3:]
        q, out = q_now, []
        for _ in range(HORIZON):
            w = w + d; d = d * a
            q = qmul(q, qexp(w * DT)); out.append(q)
        return torch.stack(out, -2)

    def forward(self, seq, positions=None):
        """seq (B, L, 4) read from a zero state → forecasts (B, P, 20, 4) after seq[:, t], t ∈ positions."""
        positions = positions if positions is not None else [seq.shape[1] - 1]
        F, keep = self.motion_matrix(), set(positions)
        state, qs, xs = self.init_state(seq.shape[0], seq.device), [], []
        for t in range(max(positions) + 1):
            x, state = self.step(seq[:, t], state, F)
            if t in keep:
                qs.append(seq[:, t]); xs.append(x)
        return self.predict(torch.stack(qs, 1), torch.stack(xs, 1))


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


def count_macs(net):
    """MACs of one streaming step (every Linear runs once) and of the 20-step forecast."""
    linear = sum(m.in_features * m.out_features for m in net.modules() if isinstance(m, nn.Linear))
    filt = 6 * 6 + 6 * 3 + 16 + 3                     # prior, gain·innovation, relative quaternion, log
    forecast = HORIZON * (16 + 14)                    # quaternion product + exp per future step
    return linear + filt, forecast


# ════════════════════════════════════ 5. export & package ════════════════════════════════════
class ContextModel(nn.Module):
    """Stateless challenge model: context [1, 400, 4] (w,x,y,z) → prediction [1, 20, 4].
    Uses the last k samples, starting from a zero state (unrolled for k steps)."""
    def __init__(self, net, k):
        super().__init__(); self.net, self.k = net, k

    def forward(self, context):
        seq = context[:, -self.k:]
        F, state = self.net.motion_matrix(), torch.zeros(1, self.net.state_size)      # batch 1, static shape
        for t in range(self.k):
            x, state = self.net.step(seq[:, t], state, F)
        pred = self.net.predict(seq[:, -1], x)
        return pred / pred.norm(dim=-1, keepdim=True)


class StreamingModel(nn.Module):
    """Stateful model: one new sample + state → prediction [1, 20, 4] + new state. Zero state = new session."""
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, q_now, state):
        x, new_state = self.net.step(q_now, state, self.net.motion_matrix())
        pred = self.net.predict(q_now, x)
        return pred / pred.norm(dim=-1, keepdim=True), new_state


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
The model is stateless: each call reads the last {k} context samples from a zero state. No network access.
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


def write_package(args, out, net, k, metrics, eff, ops, cfg):
    pkg = out / f"{args.team}_headpose"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    shutil.copy(out / "model_fp32.onnx", pkg / "model_fp32.onnx")
    shutil.copy(out / "model_streaming_fp32.onnx", pkg / "model_streaming_fp32.onnx")
    (pkg / "run_inference.py").write_text(RUN_INFERENCE.format(team=args.team, k=k))
    import onnxruntime
    (pkg / "requirements.txt").write_text(f"numpy=={np.__version__}\nonnxruntime=={onnxruntime.__version__}\n")
    res = metrics["results"]["test"]                      # final numbers: held-out TEST users
    main = res[f"KalmanNet (context {k})"]
    cv = res["constant velocity"]
    (pkg / "model_card.md").write_text(textwrap.dedent(f"""\
        # Model card — {args.team}: KalmanNet head-pose predictor

        **Task.** 400 past head orientations (unit quaternions, w,x,y,z, 100 Hz) → next 20 orientations (200 ms).

        ## Architecture
        KalmanNet (Revach et al., IEEE TSP 2022; architecture as in Song et al., IEEE SPL 2024) on a
        rotation-motion filter. Observation: head-frame angular velocity from consecutive orientations.
        State: angular velocity and its per-sample change (6). Motion model: ω ← ω + d, d ← a·d (learned decay).
        Three GRUs + fully connected layers learn the 6×3 Kalman gain from four normalised difference
        features. Prediction rolls the motion model 20 steps and composes the rotations onto the last orientation.
        GRUs are written with Linear + Sigmoid/Tanh (no recurrent ONNX operator).

        | item | value |
        |---|---|
        | trainable parameters | {eff['parameters']:,} |
        | MACs per streaming step (filter update + 20-step forecast) | {eff['macs_step'] + eff['macs_forecast']:,} |
        | MACs per call of model_fp32.onnx ({k} steps + forecast) | {eff['macs_context_call']:,} |
        | context length used | {k} samples (the last {k} of the 400 provided) |
        | state (model_streaming_fp32.onnx) | `state` float32 [1, {net.state_size}], zeros at session start; never carried across sessions |
        | quantisation / pruning / distillation | none (FP32) |
        | precision | float32 throughout |

        Files: `model_fp32.onnx` — stateless, input `context` [1, 400, 4] → output `prediction` [1, 20, 4].
        `model_streaming_fp32.onnx` — stateful alternative: inputs `q_now` [1, 4], `state` [1, {net.state_size}] →
        outputs `prediction` [1, 20, 4], `state_out` [1, {net.state_size}].

        ONNX operators: {', '.join(ops)}.

        ## Training data and preprocessing
        BOXRR-23 (Nair et al., IEEE TVCG 2024), Beat Saber head tracking, filtered and cleaned by the team's
        preprocessing (challenge filters; unit quaternions; hemisphere-aligned; 100 Hz grid; gaps split, not
        interpolated). Users split 80/10/10 (train/val/test) by a hash of the user ID;
        no user appears in two splits. {metrics['data']['train_users']:,} training users
        ({metrics['data']['train_hours']:.0f} h, at most {cfg['chunks_per_user']} chunks of {cfg['train_len']} samples
        per user), {metrics['data']['val_users']:,} validation users (model selection) and
        {metrics['data']['test_users']:,} test users (reported once, at the end). Loss: mean geodesic error at the
        20th predicted sample (+{cfg['aux']} × mean over all horizons). AdamW, one-cycle LR {cfg['lr']}, batch
        {cfg['batch']}, {cfg['steps']} steps.

        ## Test results (held-out test users, {metrics['data']['test_windows']:,} windows; never used for training or selection)
        | model | mean error @20th sample (°) | median (°) | p95 (°) |
        |---|---|---|---|
        | constant velocity | {cv['mean_200ms']:.3f} | {cv['median_200ms']:.3f} | {cv['p95_200ms']:.3f} |
        | last observation | {res['last observation']['mean_200ms']:.3f} | {res['last observation']['median_200ms']:.3f} | {res['last observation']['p95_200ms']:.3f} |
        | damped constant velocity | {res['damped CV']['mean_200ms']:.3f} | {res['damped CV']['median_200ms']:.3f} | {res['damped CV']['p95_200ms']:.3f} |
        | **KalmanNet (this model)** | **{main['mean_200ms']:.3f}** | {main['median_200ms']:.3f} | {main['p95_200ms']:.3f} |

        CPU latency (onnxruntime, 1 thread): {eff['onnx_ms_context_call']:.3f} ms per call of model_fp32.onnx,
        {eff['onnx_ms_streaming_step']:.3f} ms per streaming step.

        ## Known failure modes
        * Largest errors during fast head motion and abrupt reversals (see by_regime.csv): the forecast is a
          damped constant-acceleration roll-out, so it cannot anticipate a turn that has not started.
        * Trained and validated on Beat Saber only; other applications may move differently.
        * Accuracy assumes clean, uniformly sampled 100 Hz input; gaps or duplicated samples inside the context
          appear as sudden velocity changes.
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

        `model_fp32.onnx` is the submitted model (stateless). `model_streaming_fp32.onnx` is the same network as a
        stateful one-sample-per-call model, provided in case the harness drives models sample by sample.
        Details and metrics: `model_card.md`.
        """))
    np.save(pkg / "sample.npy", metrics.pop("_sample_context"))       # used for the test, removed before zipping
    return pkg


def zip_package(out, pkg):
    zpath = out / f"{pkg.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            if p.name != "__pycache__" and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(out))
    return zpath


# ════════════════════════════════════ main ════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="TEAMNAME", help="registered team name (used for the ZIP)")
    ap.add_argument("--data", help="folder containing users/ (skips detection / download)")
    ap.add_argument("--download-dir", default="data/headvane", help="where kagglehub downloads the corpus")
    ap.add_argument("--work", default="work", help="cache folder (fast local disk)")
    ap.add_argument("--out", default="runs/kalmannet")
    # model / training
    ap.add_argument("--context", type=int, default=100, help="samples the model reads at inference (≤ 400)")
    ap.add_argument("--span", type=int, default=100, help="training positions per chunk")
    ap.add_argument("--chunks-per-user", type=int, default=300)
    ap.add_argument("--val-windows-per-user", type=int, default=40, help="also used for test users")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of all-horizon error (0 = pure challenge metric)")
    ap.add_argument("--out-mult", type=int, default=40, help="KalmanNet FC2 width multiplier (fewer params: 10)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the pipeline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    k = args.context
    assert 2 <= k <= CONTEXT
    first = max(2, k - args.span // 2)                       # training positions centred on the inference context
    positions = list(range(first, first + args.span))
    train_len = positions[-1] + 1 + HORIZON

    # ── data ──
    root = connect(args)
    files = sorted(root.rglob("*.h5"))
    if args.smoke:
        files = random.Random(args.seed).sample(files, min(150, len(files)))
        args.chunks_per_user, args.val_windows_per_user = 50, 10
        args.steps, args.eval_every, args.log_every, args.batch = 30, 10, 5, 32
    print(f"device {device} | corpus {root} | {len(files):,} files", flush=True)
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
    def train_batch():
        idx = np.sort(rng.choice(len(TRAIN), args.batch, replace=False))
        return torch.from_numpy(np.array(TRAIN[idx]))
    val_small = WIN["val"][np.random.default_rng(1).choice(len(WIN["val"]), min(3000, len(WIN["val"])), replace=False)]

    # ── TensorBoard ──
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(out / "tensorboard")
    tb.add_text("data", "\n".join(f"* **{k_}**: {v}" for k_, v in data_info.items()))
    tb.add_text("config", "```\n" + json.dumps(vars(args), indent=2) + "\n```")

    # ── damped CV tuned on training users ──
    tune = torch.from_numpy(np.array(TRAIN[np.sort(rng.choice(len(TRAIN), min(5000, len(TRAIN)), replace=False))]))
    t_ctx, t_true = tune[:, :train_len - HORIZON], tune[:, -1]
    grid = [(b, kk) for b in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.0) for kk in (1, 2, 3, 5)]
    beta, kv = min(grid, key=lambda g: challenge_error(damped_cv(t_ctx, *g)[:, -1], t_true).mean().item())
    print(f"damped CV tuned on train users: beta={beta}, k={kv}", flush=True)
    baseline_fns = {"last observation": last_observation,
                    "constant velocity": lambda c: damped_cv(c, 1.0, 1),
                    "damped CV": lambda c: damped_cv(c, beta, kv)}
    base_val_small = {n: challenge_error(predict_windows(f, val_small, CONTEXT, device), val_small[:, CONTEXT:])[:, -1].mean().item()
                      for n, f in baseline_fns.items()}

    # ── train KalmanNet ──
    net = KalmanNet(out_mult=args.out_mult).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    tgt = (torch.tensor(positions)[:, None] + torch.arange(1, HORIZON + 1)).to(device)     # (span, 20)
    ckpt, best, hist, t0 = out / "kalmannet_best.pt", float("inf"), [], time.time()

    def kn_errors(windows, context):
        net.eval()
        e = challenge_error(predict_windows(lambda c: net(c)[:, -1], windows, context, device), windows[:, CONTEXT:])
        net.train()
        return e

    for step in range(1, args.steps + 1):
        chunk = train_batch().to(device, non_blocking=True)
        err = loss_angle_deg(net(chunk, positions), chunk[:, tgt])
        loss = err[..., -1].mean() + args.aux * err.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == 1:
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/err_20th_sample_deg", err[..., -1].mean().item(), step)
            tb.add_scalar("train/err_10th_sample_deg", err[..., 9].mean().item(), step)
            tb.add_scalar("train/err_all_horizons_deg", err.mean().item(), step)
            tb.add_scalar("train/learning_rate", sched.get_last_lr()[0], step)
            tb.add_scalar("train/grad_norm", float(grad_norm), step)
            tb.add_scalar("train/steps_per_second", step / (time.time() - t0), step)
            for ax_name, a in zip("xyz", net.decay().detach().cpu().tolist()):
                tb.add_scalar(f"model/decay_{ax_name}", a, step)
            tb.add_scalar("model/base_gain_norm", net.base_gain.norm().item(), step)
        if step % args.eval_every == 0 or step == args.steps:
            e = kn_errors(val_small, k)
            v = e[:, -1].mean().item()
            if v < best:
                best = v; torch.save(net.state_dict(), ckpt)
            tb.add_scalar("val/err_20th_sample_deg", v, step)
            tb.add_scalar("val/err_10th_sample_deg", e[:, 9].mean().item(), step)
            tb.add_scalar("val/best_err_20th_sample_deg", best, step)
            tb.add_scalars("val/compare_err_20th_sample_deg", {"KalmanNet": v, **base_val_small}, step)
            hist.append({"step": step, "train_loss": loss.item(), "val_err20": v, "seconds": time.time() - t0})
            print(f"step {step:5d}/{args.steps} | train {loss.item():.3f}° | val@20 {v:.3f}° | best {best:.3f}° "
                  f"| {time.time() - t0:.0f}s", flush=True)
            tb.flush()
    net.load_state_dict(torch.load(ckpt, map_location=device)); net.eval()
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    # ── benchmark: validation (selection) and TEST (final, used once) ──
    print("benchmarking on all validation and test windows …", flush=True)
    main_name = f"KalmanNet (context {k})"
    errors, preds_test, results = {}, {}, {}
    for split in ("val", "test"):
        W = WIN[split]
        preds = {n: predict_windows(f, W, CONTEXT, device) for n, f in baseline_fns.items()}
        for c in sorted({50, 100, 200, 400, k}):
            preds[f"KalmanNet (context {c})"] = predict_windows(lambda x: net(x)[:, -1], W, c, device)
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
        boot.append({"comparison": f"{other} − KalmanNet (test)", "ci95_low_deg": lo, "ci95_high_deg": hi,
                     "significant": bool(lo > 0)})
    pd.DataFrame(boot).to_csv(out / "bootstrap_test.csv", index=False)

    # final numbers in TensorBoard
    for split in ("val", "test"):
        for n, r in results[split].items():
            tb.add_scalar(f"final_{split}/{n.replace(' ', '_')}_err_20th_sample_deg", r["mean_200ms"], args.steps)
        for h in range(HORIZON):
            tb.add_scalars(f"final_{split}/error_vs_horizon_deg",
                           {n.replace(" ", "_"): errors[split][n][:, h].mean().item() for n in compare}, (h + 1) * 10)
        table = pd.DataFrame(results[split]).T.round(3)
        try:
            tb.add_text(f"final_{split}/results", table.to_markdown())      # needs `tabulate`
        except ImportError:
            tb.add_text(f"final_{split}/results", "```\n" + table.to_string() + "\n```")
    tb.add_text("final_test/by_regime", by_regime.round(3).to_string())
    tb.add_text("final_test/bootstrap", "\n".join(f"* {b['comparison']}: [{b['ci95_low_deg']:+.3f}, "
                                                   f"{b['ci95_high_deg']:+.3f}]°" for b in boot))

    # ── export ONNX + check ──
    print("exporting ONNX …", flush=True)
    import onnx, onnxruntime as ort
    TEST = WIN["test"]
    cpu_net = KalmanNet(out_mult=args.out_mult); cpu_net.load_state_dict(net.state_dict()); cpu_net.eval()
    ctx_model, stream_model = ContextModel(cpu_net, k).eval(), StreamingModel(cpu_net).eval()
    export_onnx(ctx_model, (TEST[:1, :CONTEXT].clone(),), out / "model_fp32.onnx", ["context"], ["prediction"])
    export_onnx(stream_model, (TEST[:1, 0].clone(), torch.zeros(1, cpu_net.state_size)),
                out / "model_streaming_fp32.onnx", ["q_now", "state"], ["prediction", "state_out"])
    ops = sorted({nd.op_type for nd in onnx.load(out / "model_fp32.onnx").graph.node} |
                 {nd.op_type for nd in onnx.load(out / "model_streaming_fp32.onnx").graph.node})
    safe = {"Conv", "MatMul", "Gemm", "LayerNormalization", "BatchNormalization", "Relu", "Gelu", "Add", "Mul",
            "Concat", "Slice", "Reshape"}
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s_ctx = ort.InferenceSession(str(out / "model_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    s_str = ort.InferenceSession(str(out / "model_streaming_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    n_chk = min(200, len(TEST))
    onnx_pred = np.concatenate([s_ctx.run(None, {"context": TEST[i:i + 1, :CONTEXT].numpy()})[0] for i in range(n_chk)])
    torch_pred = preds_test[main_name][:n_chk]
    torch_pred = (torch_pred / torch_pred.norm(dim=-1, keepdim=True)).numpy()
    state = np.zeros((1, cpu_net.state_size), np.float32)
    for t in range(CONTEXT - k, CONTEXT):
        p_stream, state = s_str.run(None, {"q_now": TEST[:1, t].numpy(), "state": state})
    checks = {"max_abs_diff_onnx_vs_torch": float(np.abs(onnx_pred - torch_pred).max()),
              "max_abs_diff_streaming_vs_context": float(np.abs(p_stream - onnx_pred[:1]).max()),
              "onnx_mean_err20_first_test_windows": challenge_error(torch.from_numpy(onnx_pred), TEST[:n_chk, CONTEXT:])[:, -1].mean().item(),
              "torch_mean_err20_first_test_windows": E[main_name][:n_chk, -1].mean().item(),
              "nan_or_inf": bool(not np.isfinite(onnx_pred).all()), "output_shape": list(onnx_pred.shape[1:])}

    def timeit(fn, n=500):
        for _ in range(20): fn()
        ts = []
        for _ in range(n):
            s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
        return float(np.median(ts) * 1e3)

    x_ctx, q1 = TEST[:1, :CONTEXT].numpy(), TEST[:1, 0].numpy()
    macs_step, macs_fc = count_macs(cpu_net)
    with torch.no_grad():
        F0, st, qq = cpu_net.motion_matrix(), cpu_net.init_state(1), TEST[:1, 0]
        torch_step_ms = timeit(lambda: cpu_net.predict(qq, cpu_net.step(qq, st, F0)[0]))
    eff = {"parameters": sum(p.numel() for p in cpu_net.parameters() if p.requires_grad),
           "macs_step": macs_step, "macs_forecast": macs_fc, "macs_context_call": k * macs_step + macs_fc,
           "context_samples": k, "state_size": cpu_net.state_size,
           "onnx_ms_context_call": timeit(lambda: s_ctx.run(None, {"context": x_ctx}), 200),
           "onnx_ms_streaming_step": timeit(lambda: s_str.run(None, {"q_now": q1, "state": state})),
           "torch_ms_streaming_step": torch_step_ms,
           "onnx_size_kb": (out / "model_fp32.onnx").stat().st_size / 1024,
           "onnx_ops": ops, "onnx_ops_outside_planning_doc_safe_list": [o for o in ops if o not in safe]}
    pd.Series({k_: v for k_, v in eff.items() if not isinstance(v, list)}).to_csv(out / "efficiency.csv")
    for k_ in ("parameters", "macs_step", "macs_context_call", "onnx_ms_context_call", "onnx_ms_streaming_step"):
        tb.add_scalar(f"efficiency/{k_}", eff[k_], args.steps)

    # ── metrics.json, config, package ──
    cfg = {**vars(args), "train_len": train_len, "train_positions": [positions[0], positions[-1]],
           "damped_cv": {"beta": beta, "k": kv}, "device": device, "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics = {"data": data_info, "results": results, "bootstrap_test": boot,
               "by_regime_test": by_regime.round(4).to_dict(),
               "efficiency": eff, "onnx_checks": checks, "best_val_err20_small": best,
               "_sample_context": TEST[0, :CONTEXT].numpy()}
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
        h = pd.DataFrame(hist); ax[0].plot(h.step, h.val_err20, marker="o", label="KalmanNet (val)")
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

    lines = [f"# KalmanNet benchmark — {args.team}", "",
             f"Train {data_info['train_users']:,} users ({data_info['train_hours']:.0f} h) · validation "
             f"{data_info['val_users']:,} users ({data_info['val_windows']:,} windows, model selection) · test "
             f"{data_info['test_users']:,} users ({data_info['test_windows']:,} windows, reported once). "
             f"Splits are user-disjoint. Metric: mean 2·acos(|q̂·q|) at the 20th sample (200 ms).", ""]
    for split in ("test", "val"):
        t = pd.DataFrame({n: r for n, r in results[split].items() if "mean_100ms" in r}).T.sort_values("mean_200ms")
        lines += [f"## {split.upper()}", "", "| model | mean @200 ms (°) | mean @100 ms (°) | median (°) | p95 (°) | vs CV |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {n} | {r.mean_200ms:.3f} | {r.mean_100ms:.3f} | {r.median_200ms:.3f} | {r.p95_200ms:.3f} | "
                  f"{r.vs_constant_velocity_pct:+.1f} % |" for n, r in t.iterrows()]
        lines += [""]
    lines += ["Per-user bootstrap on TEST (95 % CI of baseline − KalmanNet; > 0 means KalmanNet is better):", ""]
    lines += [f"* {b['comparison']}: [{b['ci95_low_deg']:+.3f}, {b['ci95_high_deg']:+.3f}]° "
              f"{'✓ significant' if b['significant'] else '· not significant'}" for b in boot]
    lines += ["", f"Parameters {eff['parameters']:,} · MACs/streaming step {eff['macs_step'] + eff['macs_forecast']:,} · "
                  f"ONNX {eff['onnx_ms_context_call']:.2f} ms/call ({k}-sample context), "
                  f"{eff['onnx_ms_streaming_step']:.3f} ms/streaming step (CPU, 1 thread)",
              f"Operators outside the planning doc's safe list: {eff['onnx_ops_outside_planning_doc_safe_list']} — "
              "check against the challenge's Submitting Artifacts page.",
              f"ONNX checks: {checks}", "", f"Package: `{zpath}`", "",
              f"TensorBoard: `tensorboard --logdir {out / 'tensorboard'}`", "", "![](fig_benchmark.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    t = pd.DataFrame({n: r for n, r in results["test"].items() if "mean_100ms" in r}).T.sort_values("mean_200ms")
    print("\nTEST (held-out users):")
    print(t[["mean_200ms", "mean_100ms", "median_200ms", "p95_200ms", "vs_constant_velocity_pct"]].round(3).to_string())
    print("\nbootstrap:", *[f"{b['comparison']}: [{b['ci95_low_deg']:+.3f}, {b['ci95_high_deg']:+.3f}]" for b in boot], sep="\n  ")
    print(f"\nparams {eff['parameters']:,} | ONNX {eff['onnx_ms_context_call']:.2f} ms/call | "
          f"package test {'OK' if checks['package_run_inference_ok'] else 'FAILED'}")
    print(f"everything saved in {out.resolve()}  |  submission: {zpath}")
    print(f"TensorBoard: tensorboard --logdir {(out / 'tensorboard').resolve()}")


if __name__ == "__main__":
    main()
