"""
probe_io.py — Save and load per-layer binding-ID probes.

Cache file is named probes_{model}_{n_probe}.joblib and stored alongside the scripts.
Scripts that use this module skip training if a matching cache already exists.
"""
import os
import joblib


def cache_path(outdir, model, n_probe):
    return os.path.join(outdir, f"probes_{model}_{n_probe}.joblib")


def save_probes(probes, path):
    joblib.dump(probes, path)
    print(f"  Saved probes → {path}")


def load_probes(path):
    probes = joblib.load(path)
    print(f"  Loaded probes ← {path}")
    return probes
