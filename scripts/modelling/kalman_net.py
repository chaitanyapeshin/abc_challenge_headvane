import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
