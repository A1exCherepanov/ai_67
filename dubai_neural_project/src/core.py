"""Leakage-controlled tabular regressors and explicit observation-error models.

Raw prices are strictly positive; all estimators learn log(P) or log(P / A).
Prediction of P from a unit-price model always uses the OBSERVED input area.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

torch.set_num_threads(2)

NUMERIC = ["bedrooms", "log_area_m2", "lat", "lon", "log_metro_min",
           "to_burj_khalifa_km", "launch_year", "handover_year", "log_project_units",
           "mortgage_rate_at_listing", "months_since_start", "month_sin", "month_cos"]
CATEGORICAL = ["developer", "developer_tier", "community", "zone", "property_category",
               "property_type", "view", "payment_plan", "metro_station", "metro_line"]
RAW_NUMERIC = ["bedrooms", "area_m2", "lat", "lon", "metro_distance_min",
               "to_burj_khalifa_km", "launch_year", "handover_year", "total_project_units",
               "mortgage_rate_at_listing"]
RAW_INPUTS = ["date_listed"] + RAW_NUMERIC + CATEGORICAL
FAMILIES = ["Median", "Ridge", "RandomForest", "LightGBM", "MLP", "TabResNet"]
TARGETS = ["total", "unit"]
SEEDS = [42, 43, 44]
LEVELS = [0.05, 0.10, 0.20]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def features(raw):
    """Stateless engineering. No target, IDs, fitted calendar origin or joins."""
    d = raw.copy()
    dt = pd.to_datetime(d["date_listed"], errors="raise")
    for col in RAW_NUMERIC:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for source, dest in [("area_m2", "log_area_m2"),
                         ("metro_distance_min", "log_metro_min"),
                         ("total_project_units", "log_project_units")]:
        values = d[source].where(d[source] > 0) if source != "metro_distance_min" else d[source].where(d[source] >= 0)
        d[dest] = np.log(values) if source == "area_m2" else np.log1p(values)
    d["months_since_start"] = (dt.dt.year - 2021) * 12 + dt.dt.month - 1
    d["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    for col in CATEGORICAL:
        d[col] = d[col].fillna("__MISSING__").astype(str)
    return d[NUMERIC + CATEGORICAL]


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              dtype=np.float32), CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)


def time_split(raw):
    dt = pd.to_datetime(raw.date_listed)
    return (raw.loc[dt < "2024-01-01"].copy(),
            raw.loc[(dt >= "2024-01-01") & (dt < "2025-01-01")].copy(),
            raw.loc[(dt >= "2025-01-01") & (dt < "2026-05-01")].copy())


def metrics(y, p, area=None):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("Nonfinite predictions")
    ape = np.abs(p - y) / y
    ans = {"n": len(y), "MAE_USD": float(mean_absolute_error(y, p)),
           "RMSE_USD": float(np.sqrt(mean_squared_error(y, p))),
           "MAPE_pct": float(100 * np.mean(ape)), "R2": float(r2_score(y, p)),
           "within20_pct": float(100 * np.mean(ape <= .2))}
    if area is not None:
        ans["MAE_USD_m2"] = float(np.mean(np.abs(y - p) / np.asarray(area)))
    return ans


class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        layers = []
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.ReLU(), nn.Dropout(dropout)]
            dim = width
        self.layers = nn.Sequential(*layers, nn.Linear(dim, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                               nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.f(x)


class TabResNet(nn.Module):
    """Own compact adaptation of the ResNet-like tabular baseline, not an exact RTDL replica."""
    def __init__(self, dim, width, blocks, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, width),
                                    *[ResidualBlock(width, dropout) for _ in range(blocks)],
                                    nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


@dataclass
class PriceModel:
    family: str
    target: str
    params: dict
    seed: int = 42

    def observed_area(self, raw):
        a = pd.to_numeric(raw.area_m2, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(a) & (a > 0), a, self.area_median)

    def _decode(self, z, raw):
        # exp is intentionally unclipped. A numerical failure must remain visible.
        p = np.exp(np.asarray(z, dtype=float))
        if self.target == "unit":
            p *= self.observed_area(raw)
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("Nonfinite price; inspect training rather than clip output")
        return p

    def fit(self, train, val, max_epochs=220, patience=25):
        seed_all(self.seed)
        start = time.perf_counter()
        self.area_median = float(pd.to_numeric(train.area_m2).median())
        y = np.log(train.price_usd.to_numpy(dtype=float))
        if self.target == "unit":
            y -= np.log(self.observed_area(train))
        self.prep = preprocessor()
        x = self.prep.fit_transform(features(train)).astype(np.float32)
        v = self.prep.transform(features(val)).astype(np.float32)
        self.input_dim = x.shape[1]
        self.history = []
        self.best_iteration = None
        if self.family == "Median":
            self.model = float(np.median(y))
        elif self.family == "Ridge":
            self.model = Ridge(**self.params).fit(x, y)
        elif self.family == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=220, n_jobs=2,
                       random_state=self.seed, **self.params).fit(x, y)
        elif self.family == "LightGBM":
            self.model = LGBMRegressor(objective="regression", metric="None", n_estimators=2500,
                        learning_rate=.03, colsample_bytree=.9, reg_lambda=1.,
                        random_state=self.seed, n_jobs=2, verbosity=-1, deterministic=True,
                        force_col_wise=True, **self.params)
            yv = np.log(val.price_usd.to_numpy(dtype=float))
            if self.target == "unit":
                yv -= np.log(self.observed_area(val))
            def valid_mae(_target, pred):
                return "price_mae", float(np.mean(np.abs(self._decode(pred, val) - val.price_usd))), False
            self.model.fit(x, y, eval_set=[(v, yv)], eval_metric=valid_mae,
                callbacks=[early_stopping(100, first_metric_only=True, verbose=False), log_evaluation(0)])
            self.best_iteration = int(self.model.best_iteration_)
        else:
            self.y_mean, self.y_std = float(y.mean()), float(y.std())
            yy = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
            xt, vt = torch.from_numpy(x), torch.from_numpy(v)
            config = dict(self.params)
            lr = config.pop("lr", .001)
            decay = config.pop("weight_decay", .0001)
            self.model = (MLP(x.shape[1], **config) if self.family == "MLP"
                          else TabResNet(x.shape[1], **config))
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)
            best, wait, state = np.inf, 0, None
            batch = 256
            for epoch in range(1, max_epochs + 1):
                self.model.train()
                order = torch.randperm(len(xt))
                loss_total = 0.
                for idx in order.split(batch):
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((self.model(xt[idx]) - yy[idx]) ** 2).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    optimizer.step()
                    loss_total += loss.item() * len(idx)
                self.model.eval()
                with torch.no_grad():
                    zv = self.model(vt).numpy() * self.y_std + self.y_mean
                mae = float(np.mean(np.abs(self._decode(zv, val) - val.price_usd)))
                self.history.append({"epoch": epoch, "train_standardized_MSE": loss_total / len(xt),
                                     "val_MAE_USD": mae})
                if mae < best:
                    best, wait = mae, 0
                    state = copy.deepcopy(self.model.state_dict())
                    self.best_iteration = epoch
                else:
                    wait += 1
                    if wait >= patience:
                        break
            self.model.load_state_dict(state)
            self.model.eval()
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict_log(self, raw):
        if self.family == "Median":
            return np.full(len(raw), self.model)
        x = self.prep.transform(features(raw)).astype(np.float32)
        if self.family in ["MLP", "TabResNet"]:
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.from_numpy(x)).numpy() * self.y_std + self.y_mean
        return self.model.predict(x)

    def predict(self, raw):
        return self._decode(self.predict_log(raw), raw)


def candidates(family):
    return {
        "Median": [{}],
        "Ridge": [{"alpha": a} for a in [1., 10., 100.]],
        "RandomForest": [{"min_samples_leaf": m, "max_features": .9} for m in [1, 3, 5]],
        "LightGBM": [{"num_leaves": n, "min_child_samples": 20} for n in [15, 31, 63]],
        "MLP": [dict(hidden=h, dropout=d, lr=.001, weight_decay=.0001)
                for h,d in [((64,32),.05), ((128,64,32),.05), ((128,64,32),.15)]],
        "TabResNet": [dict(width=w, blocks=b, dropout=d, lr=.001, weight_decay=.0001)
                      for w,b,d in [(64,2,.05),(128,2,.05),(64,3,.10)]],
    }[family]


def corrupt(raw, kind, share, seed, train_pools=None):
    """Nested masks across shares; levels denote selected rows per affected field."""
    out = raw.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(out)
    k = int(round(share * n))
    order = rng.permutation(n)
    factors = rng.uniform(.7, 1.3, n)
    if kind == "area":
        out.loc[order[:k], "area_m2"] *= factors[order[:k]]
        if "area_sqft" in out:
            out["area_sqft"] = out.area_m2 / .09290304
    elif kind == "view":
        pool = sorted(train_pools["view"])
        for i in order[:k]:
            choices = [p for p in pool if p != out.at[i, "view"]]
            out.at[i, "view"] = rng.choice(choices)
    elif kind == "missing":
        for col in ["area_m2", "metro_distance_min"]:
            ids = rng.permutation(n)[:k]
            out.loc[ids, col] = np.nan
        if "area_sqft" in out:
            out["area_sqft"] = out.area_m2 / .09290304
    elif kind == "price":
        # Unlike area noise, the label experiment uses 0.5..1.5.
        factors = rng.uniform(.5, 1.5, n)
        out.loc[order[:k], "price_usd"] *= factors[order[:k]]
    else:
        raise ValueError(kind)
    return out


def load_data(path):
    d = pd.read_csv(path)
    d["date_listed"] = pd.to_datetime(d.date_listed)
    for c in ["area_m2", "area_sqft", "price_usd"]:
        d[c] = d[c].astype(float)
    # Fixed domain rules only; never fit price thresholds on a holdout.
    valid = d.date_listed.notna() & np.isfinite(d.price_usd) & (d.price_usd > 0) & (d.area_m2 > 0)
    if not valid.all():
        raise ValueError("Source has invalid dates/prices/areas; inspect explicitly before changing the benchmark")
    if d.id.duplicated().any():
        raise ValueError("Duplicate listing IDs")
    return d.reset_index(drop=True)
