"""
Random Forest (modelo (a) del enunciado) con el mismo protocolo que las redes: mismo dataset, mismo split por
documento (src/protocolo.py) y misma métrica. Corre en CPU. Son tres bosques independientes, uno por tarea.

    --modelo con_emb   entrada = embedding de la palabra (768 dim, promedio de sus tokens BERT)   [a-i]
    --modelo sin_emb   entrada = atributos "a la antigua" (largo, vocales, acentos, vecinas...)   [a-ii]

Procedimiento:
  1. Búsqueda: cada combinación de la grilla (class_weight, max_depth, min_samples_leaf) entrena un bosque chico
     sobre una muestra de grupos del train y se mide en VAL. Se elige por tarea la de mayor F1 macro.
  2. Final: se entrena el bosque grande con TODO el train, se ajustan factores por clase sobre las
     probabilidades con VAL (como la etapa D de las redes: se usan solo si la ganancia medida en la otra mitad
     de val es >= --min-ganancia-umbral) y se mide en TEST una sola vez.
  3. Se guardan las predicciones de test (preds_test.csv) para src/evaluacion.py.

Uso:
    python src/busqueda_rf.py --modelo sin_emb
    python src/busqueda_rf.py --modelo con_emb
    python src/busqueda_rf.py --modelo sin_emb --rapido        # prueba de humo
Es reanudable: las corridas de búsqueda ya hechas (mismo dataset y split) no se repiten.
"""
import argparse
import itertools
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import atributos_rf
from protocolo import (K_CLASES, NOMBRES, asignar_grupo, buscar_factores, cargar_etiquetas, dividir, f1_macro,
                       guardar_predicciones, huella_datos, huella_split, reporte_split)
from rutas import SALIDAS

GRILLA = {"class_weight": [None, "balanced"], "max_depth": [12, 20, 30], "min_samples_leaf": [1, 3]}
GRILLA_RAPIDO = {"class_weight": [None, "balanced"], "max_depth": [12], "min_samples_leaf": [1]}
CAMPOS = ["class_weight", "max_depth", "min_samples_leaf"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def combinaciones(grilla):
    return [dict(zip(CAMPOS, v)) for v in itertools.product(*(grilla[c] for c in CAMPOS))]


def proba_completa(rf, X, K):
    """predict_proba con las K columnas (si a un bosque le faltó una clase en el entrenamiento, queda en 0)."""
    P = np.zeros((X.shape[0], K))
    P[:, rf.classes_] = rf.predict_proba(X)
    return P


class Registro:
    """CSV de corridas de la búsqueda; no repite las que ya están (mismo dataset, split y configuración)."""

    def __init__(self, csv, clave):
        self.csv, self.clave = csv, clave

    def _fila_id(self, tarea, cfg, n_arboles):
        return {"tarea": tarea, "class_weight": str(cfg["class_weight"]), "max_depth": cfg["max_depth"],
                "min_samples_leaf": cfg["min_samples_leaf"], "n_arboles": n_arboles}

    def buscar(self, tarea, cfg, n_arboles):
        if not self.csv.exists():
            return None
        d = pd.read_csv(self.csv)
        m = d.clave == self.clave
        for k, v in self._fila_id(tarea, cfg, n_arboles).items():
            m &= d[k].astype(str) == str(v)
        return d[m].iloc[-1] if m.any() else None

    def agregar(self, tarea, cfg, n_arboles, f1_val, segundos):
        fila = {"clave": self.clave, **self._fila_id(tarea, cfg, n_arboles), "f1_val": round(f1_val, 4),
                "segundos": round(segundos)}
        pd.DataFrame([fila]).to_csv(self.csv, mode="a", header=not self.csv.exists(), index=False)
        return pd.Series(fila)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, choices=["con_emb", "sin_emb"])
    ap.add_argument("--arboles-busqueda", type=int, default=30)
    ap.add_argument("--arboles-final", type=int, default=150)
    ap.add_argument("--frac-busqueda", type=float, default=0.25, help="fracción de los grupos de train que usa la búsqueda")
    ap.add_argument("--semillas", type=int, default=1, help="semillas del bosque final (media y desvío en test)")
    ap.add_argument("--min-ganancia-umbral", type=float, default=0.005)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--guardar-modelo", action="store_true", help="guarda los bosques (con_emb pesan ~1 GB)")
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos datos y árboles")
    a = ap.parse_args()

    carpeta = SALIDAS / f"rf_{a.modelo}{'_rapido' if a.rapido else ''}"
    carpeta.mkdir(exist_ok=True)
    csv, npy = SALIDAS / "etiquetas_palabras.csv", SALIDAS / "embeddings_palabras.npy"
    et = cargar_etiquetas(csv)
    E = np.load(npy, mmap_mode="r")
    assert len(E) == len(et), "embeddings y etiquetas no son del mismo run"
    huella = huella_datos(csv, E)
    idx_tr, idx_val, idx_te = dividir(et)
    h_split = huella_split(idx_tr, idx_val, idx_te)
    (SALIDAS / "split_reporte.txt").write_text(reporte_split(et, idx_tr, idx_val, idx_te), encoding="utf-8")
    log(f"rf {a.modelo} | dataset {huella} | split {h_split} | palabras train/val/test "
        f"{len(idx_tr):,}/{len(idx_val):,}/{len(idx_te):,}")

    vocab = None
    if a.modelo == "con_emb":
        def feats(idx):
            return np.asarray(E[idx], dtype=np.float32)
    else:
        df = pd.DataFrame({"instancia_id": et["instancia_id"].to_numpy(), "_norm": et["palabra"].str.lower().str.strip()})
        vocab = [w for w, _ in Counter(df["_norm"].to_numpy()[idx_tr]).most_common(atributos_rf.TOP_K_PALABRAS)]  # solo train
        log("construyendo atributos...")
        F = atributos_rf.construir_atributos(df, vocab)
        log(f"atributos: {F.shape}")

        def feats(idx):
            return F[idx]

    # --- muestra de grupos de train para la búsqueda
    grupo = asignar_grupo(et)
    rng = np.random.default_rng(0)
    frac = 0.03 if a.rapido else a.frac_busqueda
    g_tr = np.unique(grupo[idx_tr])
    elegidos = set(rng.choice(g_tr, size=max(2, int(len(g_tr) * frac)), replace=False))
    idx_busq = idx_tr[np.isin(grupo[idx_tr], list(elegidos))]
    n_arb_busq, n_arb_final = (8, 10) if a.rapido else (a.arboles_busqueda, a.arboles_final)
    grilla = GRILLA_RAPIDO if a.rapido else GRILLA
    log(f"búsqueda: {len(idx_busq):,} palabras de train ({len(elegidos)} grupos), {n_arb_busq} árboles, "
        f"{len(combinaciones(grilla))} combinaciones por tarea")

    registro = Registro(carpeta / "busqueda_rf.csv", f"{huella}-{h_split}")
    Xb, Xv = feats(idx_busq), feats(idx_val)
    mejores = {}
    for tarea in NOMBRES:
        K, y = K_CLASES[tarea], et[tarea + "_id"].to_numpy()
        candidatos = []
        for cfg in combinaciones(grilla):
            fila = registro.buscar(tarea, cfg, n_arb_busq)
            marca = " (ya hecha)"
            if fila is None:
                t0 = time.time()
                rf = RandomForestClassifier(n_estimators=n_arb_busq, max_features="sqrt", n_jobs=a.n_jobs,
                                            random_state=0, **cfg)
                rf.fit(Xb, y[idx_busq])
                f1 = f1_macro(y[idx_val], proba_completa(rf, Xv, K).argmax(1), K)
                fila, marca = registro.agregar(tarea, cfg, n_arb_busq, f1, time.time() - t0), ""
            candidatos.append((float(fila.f1_val), cfg))
            log(f"  {tarea:15s} {cfg} -> F1 val {float(fila.f1_val):.4f}{marca}")
        mejores[tarea] = max(candidatos, key=lambda c: c[0])[1]
        log(f"  => {tarea}: {mejores[tarea]}")
    del Xb

    # --- final: bosque grande con todo train, factores con val, test una sola vez
    Xt, Xte = feats(idx_tr), feats(idx_te)
    g_val = grupo[idx_val]
    mitad_A = np.isin(g_val, np.random.default_rng(0).permutation(np.unique(g_val))[: len(np.unique(g_val)) // 2])
    filas, preds_mejor, modelos = [], {}, {}
    for seed in range(a.semillas):
        fila = {"dataset": huella, "split": h_split, "modelo": a.modelo, "seed": seed, "n_arboles": n_arb_final}
        for tarea in NOMBRES:
            K, y, cfg = K_CLASES[tarea], et[tarea + "_id"].to_numpy(), mejores[tarea]
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=n_arb_final, max_features="sqrt", n_jobs=a.n_jobs,
                                        random_state=seed, **cfg)
            rf.fit(Xt, y[idx_tr])
            Pv, Pte = proba_completa(rf, Xv, K), proba_completa(rf, Xte, K)
            yv = y[idx_val]
            ganancia = np.mean([
                f1_macro(yv[med], (Pv[med] * buscar_factores(Pv[opt], yv[opt], K)).argmax(1), K)
                - f1_macro(yv[med], Pv[med].argmax(1), K)
                for opt, med in [(mitad_A, ~mitad_A), (~mitad_A, mitad_A)]])
            usar = bool(ganancia >= a.min_ganancia_umbral)
            w = buscar_factores(Pv, yv, K) if usar else np.ones(K)
            f1_val = f1_macro(yv, (Pv * w).argmax(1), K)
            f1_sin, f1_con = f1_macro(y[idx_te], Pte.argmax(1), K), f1_macro(y[idx_te], (Pte * w).argmax(1), K)
            fila.update({f"val_{tarea}": round(f1_val, 4), f"test_{tarea}": round(f1_sin, 4),
                         f"test_umbral_{tarea}": round(f1_con, 4), f"usa_umbral_{tarea}": usar,
                         f"config_{tarea}": json.dumps(cfg)})
            log(f"  final s{seed} {tarea:15s} {cfg} | TEST F1 {f1_sin:.4f}"
                + (f" -> con factores {f1_con:.4f}" if usar else "") + f" | {time.time() - t0:.0f} s")
            if tarea not in preds_mejor or f1_val > preds_mejor[tarea][0]:      # la mejor semilla por VAL, no por test
                preds_mejor[tarea] = (f1_val, (Pte * w).argmax(1), seed, usar, w.tolist())
            if a.guardar_modelo:
                modelos[(tarea, seed)] = rf
        fila["score_test"] = round(np.mean([fila[f"test_{t}"] for t in NOMBRES]), 4)
        fila["score_test_umbral"] = round(np.mean([fila[f"test_umbral_{t}"] for t in NOMBRES]), 4)
        filas.append(fila)
    df_f = pd.DataFrame(filas)
    ruta_final = carpeta / "resultados_finales.csv"
    df_f.to_csv(ruta_final, mode="a", header=not ruta_final.exists(), index=False)
    cols = ([f"test_{t}" for t in NOMBRES] + ["score_test"] + [f"test_umbral_{t}" for t in NOMBRES]
            + ["score_test_umbral"])
    log("resumen en TEST (media y desvío entre semillas):")
    print(pd.DataFrame({"media": df_f[cols].mean(), "desvío": df_f[cols].std(ddof=0)}).round(3).to_string())

    ruta = guardar_predicciones(
        carpeta, f"rf_{a.modelo}", idx_te, {t: preds_mejor[t][1] for t in NOMBRES}, huella,
        {"split": h_split, "semilla_por_tarea": {t: preds_mejor[t][2] for t in NOMBRES},
         "usa_umbral": {t: preds_mejor[t][3] for t in NOMBRES}, "factores": {t: preds_mejor[t][4] for t in NOMBRES},
         "configs": mejores})
    (carpeta / "estado_busqueda.json").write_text(json.dumps({huella: {"configs": mejores}}, indent=1, default=str),
                                                  encoding="utf-8")
    log(f"predicciones de test guardadas en {ruta}")
    if a.guardar_modelo:
        import joblib
        # solo el bosque elegido de cada tarea (el de la mejor semilla por val), con sus factores por clase:
        # es lo que necesita src/inferir.py para reproducir exactamente las predicciones de preds_test.csv
        joblib.dump({"modelo": a.modelo, "dataset": huella,
                     "bosques": {t: modelos[(t, preds_mejor[t][2])] for t in NOMBRES},
                     "factores": {t: preds_mejor[t][4] for t in NOMBRES},
                     "semilla_por_tarea": {t: preds_mejor[t][2] for t in NOMBRES},
                     "configs": mejores, "vocab": vocab}, carpeta / "modelo_rf.joblib", compress=3)
        log("modelo guardado en", carpeta / "modelo_rf.joblib")


if __name__ == "__main__":
    main()
