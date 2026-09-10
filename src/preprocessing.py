"""
Preprocessing stage of the framework:

    Network Traffic data -> Data cleaning -> Missing-value handling
    -> Categorical feature encoding -> Feature normalization

Implements the exact steps named in the source paper's methodology
section, using real scikit-learn transforms (no hand-rolled stand-ins)
against the real NSL-KDD feature set (41 traffic features).
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
ARTIFACT_DIR = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Official 41 NSL-KDD feature names, in file order, plus the trailing
# label / difficulty columns.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
ALL_COLUMNS = FEATURE_COLUMNS + ["attack_type", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]

# NSL-KDD attack_type -> coarse attack category, per the standard mapping
# used across NSL-KDD literature (DoS / Probe / R2L / U2R / normal).
ATTACK_CATEGORY = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "mailbomb": "dos", "processtable": "dos", "udpstorm": "dos",
    "apache2": "dos", "worm": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD CSV (no header) into a labeled DataFrame."""
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning + missing-value handling.

    NSL-KDD ships with no missing cells, but real traffic captures do, so
    this stays generic: drop exact-duplicate rows (a known NSL-KDD
    cleanliness step), coerce numeric columns, and impute any residual
    NaNs with the column median (robust to the heavy-tailed byte/rate
    features) rather than silently dropping rows.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[NUMERIC].isna().any().any():
        df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median(numeric_only=True))
    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    """normal -> 0, any attack_type -> 1 (ground truth, evaluation-only)."""
    return (df["attack_type"].astype(str) != "normal").astype(int).to_numpy()


def category_labels(df: pd.DataFrame) -> np.ndarray:
    return df["attack_type"].astype(str).map(
        lambda a: ATTACK_CATEGORY.get(a, "unknown")
    ).to_numpy()


class Preprocessor:
    """Categorical encoding (one-hot) + feature normalization (z-score),
    fit once on the training split and reused everywhere else -- exactly
    the fit/transform discipline a real deployment needs so the RL agent
    and the API see identically-scaled features.
    """

    def __init__(self):
        self.ct = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
                ("num", StandardScaler(), NUMERIC),
            ]
        )
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        self.ct.fit(df[FEATURE_COLUMNS])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor.fit() must be called first")
        Z = self.ct.transform(df[FEATURE_COLUMNS]).astype(np.float32)
        # NSL-KDD has several near-constant raw columns (e.g. num_outbound_cmds,
        # urgent, is_host_login); a single rare non-zero row there produces a
        # huge z-score after StandardScaler. Winsorize to +-5 sigma -- a
        # standard robust-scaling step -- so K-Means centroids and the PCA
        # projection aren't dragged around by one extreme cell (one-hot
        # columns are 0/1 and unaffected by this clip).
        np.clip(Z, -5.0, 5.0, out=Z)
        return Z

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str = None):
        path = path or os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
        joblib.dump(self.ct, path)
        return path

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
        obj = cls()
        obj.ct = joblib.load(path)
        obj.fitted = True
        return obj


def load_and_prepare(split: str = "train"):
    """Convenience end-to-end loader: raw file -> cleaned DataFrame
    -> binary/category labels. Does NOT fit/apply scaling (callers decide
    whether to fit a fresh Preprocessor or reuse a saved one).
    """
    fname = "KDDTrain+.txt" if split == "train" else "KDDTest+.txt"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )
    df = load_raw(path)
    df = clean(df)
    y_bin = binary_labels(df)
    y_cat = category_labels(df)
    return df, y_bin, y_cat


if __name__ == "__main__":
    df, y_bin, y_cat = load_and_prepare("train")
    print(f"rows={len(df)}  attack_rate={y_bin.mean():.3f}")
    pre = Preprocessor()
    Z = pre.fit_transform(df)
    print(f"encoded feature matrix: {Z.shape}")
    print("category breakdown:", pd.Series(y_cat).value_counts().to_dict())
