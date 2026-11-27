"""95 % confidence intervals for the test results, resampling whole documents.

Words from the same film, briefing or sentence are not independent, so the bootstrap draws the split's own groups
(see src/protocolo.py) with replacement instead of single words. Uses the committed test predictions
(results/final/<model>/preds_test.csv) and the labels you rebuild locally (salidas/etiquetas_palabras.csv).

    python scripts/bootstrap_ci.py                 # 2000 resamples, writes results/final/bootstrap_ci.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from protocolo import K_CLASES, NOMBRES, asignar_grupo, cargar_etiquetas, dividir  # noqa: E402

MODELS = ["rf_sin_emb", "rf_con_emb", "lstm", "bilstm"]


def confusion_by_group(group_ids, y, pred, n_groups, k):
    """(n_groups, k, k) word counts."""
    flat = np.bincount(group_ids * k * k + y * k + pred, minlength=n_groups * k * k)
    return flat.reshape(n_groups, k, k)


def macro_f1_from_confusion(conf):
    """conf (..., k, k) -> macro F1 over the k classes (0 for a class with no support and no predictions)."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(axis=-1) + conf.sum(axis=-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(axis=-1)


def bootstrap(group_ids, labels, preds, n_boot, seed=0):
    """labels: {task: y}; preds: {model: {task: pred}}. Returns {model: (n_boot,) mean macro F1} and point values."""
    n_groups = int(group_ids.max()) + 1
    weights = np.random.default_rng(seed).multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    samples, point = {}, {}
    for model, by_task in preds.items():
        s, p = [], []
        for task, pred in by_task.items():
            k = K_CLASES[task]
            conf = confusion_by_group(group_ids, labels[task], pred, n_groups, k)
            p.append(macro_f1_from_confusion(conf.sum(axis=0)))
            totals = (weights @ conf.reshape(n_groups, k * k)).reshape(n_boot, k, k)
            s.append(macro_f1_from_confusion(totals))
        samples[model] = np.mean(s, axis=0)
        point[model] = float(np.mean(p))
    return samples, point


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(ROOT / "salidas" / "etiquetas_palabras.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    et = cargar_etiquetas(a.labels)
    _, _, idx_test = dividir(et)
    labels, preds = None, {}
    for model in MODELS:
        df = pd.read_csv(ROOT / "results" / "final" / model / "preds_test.csv")
        assert np.array_equal(df["fila"].to_numpy(), idx_test), (
            "the rebuilt test split does not match the committed predictions "
            "(check the data files and that scikit-learn is 1.6 or 1.7, see README)")
        preds[model] = {t: df["pred_" + t].to_numpy() for t in NOMBRES}
    labels = {t: et[t + "_id"].to_numpy()[idx_test].astype(int) for t in NOMBRES}
    _, group_ids = np.unique(asignar_grupo(et)[idx_test], return_inverse=True)

    samples, point = bootstrap(group_ids, labels, preds, a.n_boot)
    rows = []
    for model in MODELS:
        lo, hi = np.percentile(samples[model], [2.5, 97.5])
        rows.append({"comparison": model, "mean_macro_f1": point[model], "ci_low": lo, "ci_high": hi})
    for other in MODELS[:-1]:
        diff = samples["bilstm"] - samples[other]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"comparison": f"bilstm - {other}", "mean_macro_f1": point["bilstm"] - point[other],
                     "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(ROOT / "results" / "final" / "bootstrap_ci.csv", index=False)
    print(f"{group_ids.max() + 1} test groups, {len(idx_test):,} words, {a.n_boot} resamples")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
