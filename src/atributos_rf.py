"""Atributos "a la antigua" (sin embeddings) para el Random Forest: largo, vocales, acentos, dígitos, posición,
listas de palabras (interrogativos, conectores, artículos) y las mismas cosas de las 2 palabras vecinas a cada lado,
más un one-hot de las palabras más frecuentes del train.

Código de Santi (notebooks/random_forest_sin_emb.ipynb), extraído acá para usarlo desde src/busqueda_rf.py.
Entrada: un DataFrame con `instancia_id` (filas contiguas) y `_norm` (la palabra en minúsculas).
"""
import numpy as np
import pandas as pd
from scipy import sparse

VENTANA = 2
TOP_K_PALABRAS = 150

INTERROGATIVOS = {
    "que", "qué", "quien", "quién", "quienes", "quiénes", "cual", "cuál",
    "cuales", "cuáles", "cuando", "cuándo", "donde", "dónde", "como", "cómo",
    "cuanto", "cuánto", "cuanta", "cuánta", "cuantos", "cuántos", "cuantas", "cuántas",
}
CONECTORES_COMA = {
    "pero", "aunque", "sino", "ademas", "además", "entonces", "embargo",
    "osea", "decir", "ejemplo", "porque", "mientras", "pues",
}
ARTICULOS_PREP = {"el", "la", "los", "las", "un", "una", "de", "del", "en", "a", "y", "o"}
VOCALES = set("aeiouáéíóúü")


def atributos_palabra(p: str, pos: int, total: int) -> list:
    largo = len(p)
    vocales = sum(c in VOCALES for c in p)
    return [
        largo, vocales,
        sum(c.isalpha() and c not in VOCALES for c in p),
        sum(c in "áéíóú" for c in p),
        sum(c.isdigit() for c in p),
        int("'" in p or "’" in p),
        int("-" in p),
        int(p.isalpha()),
        vocales / max(largo, 1),
        int(p in INTERROGATIVOS),
        int(p in CONECTORES_COMA),
        int(p in ARTICULOS_PREP),
        int(pos == 0),
        int(pos == total - 1),
        pos / max(total - 1, 1),
    ]


def desplazar(M: np.ndarray, off: int, inst: np.ndarray, fill):
    """M[i+off] si pertenece a la misma instancia que i; si no, `fill`."""
    n = len(M)
    idx = np.arange(n) + off
    ok = (idx >= 0) & (idx < n)
    idx = np.clip(idx, 0, n - 1)
    ok &= inst[idx] == inst
    out = M[idx].copy()
    out[~ok] = fill
    return out


def construir_atributos(df: pd.DataFrame, vocab: list) -> sparse.csr_matrix:
    """df: columnas `instancia_id` (contiguas) y `_norm`. Devuelve matriz esparsa."""
    inst = df["instancia_id"].to_numpy()
    palabras = df["_norm"].tolist()
    n = len(df)
    pos = df.groupby("instancia_id", sort=False).cumcount().to_numpy()
    total = df.groupby("instancia_id", sort=False)["_norm"].transform("size").to_numpy()
    base = np.array([atributos_palabra(p, int(i), int(t))
                     for p, i, t in zip(palabras, pos, total)], dtype=np.float32)

    idx_vocab = {w: i for i, w in enumerate(vocab)}
    cod = np.array([idx_vocab.get(p, -1) for p in palabras], dtype=np.int64)
    K = len(vocab)

    densos, lexicos = [], []
    for off in range(-VENTANA, VENTANA + 1):
        densos.append(desplazar(base, off, inst, -1.0))
        sh = desplazar(cod, off, inst, -1)
        filas = np.where(sh >= 0)[0]
        lexicos.append(sparse.csr_matrix(
            (np.ones(len(filas), dtype=np.float32), (filas, sh[filas])), shape=(n, K)))
    X_dense = sparse.csr_matrix(np.hstack(densos))
    return sparse.hstack([X_dense] + lexicos, format="csr", dtype=np.float32)
