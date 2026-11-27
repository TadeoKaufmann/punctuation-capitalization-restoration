"""
Evaluación final: compara los modelos con las predicciones de TEST que guardó cada uno (preds_test.csv).

Para cada modelo y cada tarea calcula, a nivel palabra (como pide el enunciado):
  - F1 macro y F1 por clase (+ precisión y recall por clase)
  - matriz de confusión (normalizada por fila: cada fila suma 1 = el recall de esa clase real)
  - F1 macro por fuente (conferencias, subtitulos, ...) y por largo de instancia (para ver dónde falla)
y arma las figuras y tablas en salidas/evaluacion/.

Los modelos son las carpetas de salidas/ que tienen un preds_test.csv (rf_sin_emb, rf_con_emb, lstm, bilstm).
Se evalúa solo con palabras de test, y solo con las que TODOS los modelos predijeron (para que sea comparable).

Uso:  python src/evaluacion.py            (o `import evaluacion; evaluacion.evaluar()` desde el notebook)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from protocolo import CLASES, K_CLASES, NOMBRES, cargar_etiquetas, f1_macro, matriz_confusion
from rutas import SALIDAS

MODELOS = {"rf_sin_emb": "RF sin embeddings", "rf_con_emb": "RF con embeddings", "lstm": "LSTM", "bilstm": "BiLSTM"}
TAREAS_LEGIBLES = {"punt_inicial": "Puntuación inicial", "punt_final": "Puntuación final", "capitalizacion": "Capitalización"}
BINS_LARGO = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)]          # palabras de la instancia


def cargar_preds(salidas=SALIDAS) -> dict:
    """{carpeta: DataFrame(fila, pred_*)} de los modelos que ya tienen predicciones de test."""
    out = {}
    for carpeta in MODELOS:
        ruta = Path(salidas) / carpeta / "preds_test.csv"
        if ruta.exists():
            out[carpeta] = pd.read_csv(ruta)
    return out


def _largo_instancia(et: pd.DataFrame) -> np.ndarray:
    return et.groupby("instancia_id")["posicion"].transform("size").to_numpy()


def evaluar(salidas=SALIDAS, salida=None, mostrar=True):
    salidas = Path(salidas)
    salida = Path(salida) if salida else salidas / "evaluacion"
    salida.mkdir(parents=True, exist_ok=True)
    et = cargar_etiquetas(salidas / "etiquetas_palabras.csv")
    preds = cargar_preds(salidas)
    if not preds:
        raise SystemExit("no hay ningún preds_test.csv: correr primero la etapa final de algún modelo")

    comunes = sorted(set.intersection(*(set(p["fila"]) for p in preds.values())))
    filas = np.asarray(comunes)
    print(f"modelos: {', '.join(MODELOS[m] for m in preds)} | palabras de test en común: {len(filas):,}")
    real = {n: et[n + "_id"].to_numpy()[filas] for n in NOMBRES}
    fuente = et["fuente"].to_numpy()[filas]
    largo = _largo_instancia(et)[filas]
    pred = {m: {n: p.set_index("fila").loc[filas, "pred_" + n].to_numpy() for n in NOMBRES} for m, p in preds.items()}

    # --- F1 macro por modelo y tarea
    resumen = pd.DataFrame({MODELOS[m]: {TAREAS_LEGIBLES[n]: f1_macro(real[n], pred[m][n], K_CLASES[n]) for n in NOMBRES}
                            for m in preds})
    resumen.loc["Promedio"] = resumen.mean()
    resumen.round(4).to_csv(salida / "f1_macro.csv")

    # --- F1, precisión y recall por clase
    filas_clase = []
    for m in preds:
        for n in NOMBRES:
            p, r, f, soporte = precision_recall_fscore_support(real[n], pred[m][n], labels=list(range(K_CLASES[n])),
                                                               zero_division=0)
            for k, nombre in enumerate(CLASES[n]):
                filas_clase.append({"modelo": MODELOS[m], "tarea": TAREAS_LEGIBLES[n], "clase": nombre,
                                    "palabras": int(soporte[k]), "precision": p[k], "recall": r[k], "f1": f[k]})
    por_clase = pd.DataFrame(filas_clase)
    por_clase.round(4).to_csv(salida / "f1_por_clase.csv", index=False)

    # --- por fuente y por largo
    def f1_en(mascara):
        return pd.DataFrame({MODELOS[m]: {TAREAS_LEGIBLES[n]: f1_macro(real[n][mascara], pred[m][n][mascara], K_CLASES[n])
                                          for n in NOMBRES} for m in preds})
    por_fuente = pd.concat({f"{f} ({int((fuente == f).sum()):,} pal.)": f1_en(fuente == f) for f in sorted(set(fuente))})
    por_fuente.round(4).to_csv(salida / "f1_por_fuente.csv")
    por_largo = pd.concat({f"{a}-{b if b < 10_000 else '+'} pal. ({int(((largo >= a) & (largo <= b)).sum()):,})":
                           f1_en((largo >= a) & (largo <= b)) for a, b in BINS_LARGO
                           if ((largo >= a) & (largo <= b)).sum() > 0})
    por_largo.round(4).to_csv(salida / "f1_por_largo.csv")

    # --- figuras
    _figura_f1_clase(por_clase, salida)
    _figuras_confusion(preds.keys(), real, pred, salida)
    _figura_resumen(resumen, salida)

    if mostrar:
        pd.set_option("display.width", 200)
        print("\nF1 macro en test (a nivel palabra):\n", resumen.round(3).to_string())
        print("\nF1 por clase:\n", por_clase.pivot_table(index=["tarea", "clase"], columns="modelo", values="f1").round(3).to_string())
        print("\nF1 macro por fuente:\n", por_fuente.round(3).to_string())
        print("\nF1 macro por largo de instancia:\n", por_largo.round(3).to_string())
    print(f"\ntablas y figuras en {salida}")
    return {"resumen": resumen, "por_clase": por_clase, "por_fuente": por_fuente, "por_largo": por_largo}


def _figura_resumen(resumen, salida):
    ax = resumen.plot.bar(figsize=(9, 4), rot=0, width=0.8)
    ax.set_ylabel("F1 macro (test, por palabra)")
    ax.set_ylim(0, 1)
    ax.set_title("Comparación de modelos")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(salida / "comparacion_f1_macro.png", dpi=150)
    plt.close()


def _figura_f1_clase(por_clase, salida):
    for n in NOMBRES:
        d = por_clase[por_clase.tarea == TAREAS_LEGIBLES[n]].pivot(index="clase", columns="modelo", values="f1")
        d = d.reindex(CLASES[n])
        ax = d.plot.bar(figsize=(8, 3.8), rot=0, width=0.8)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"{TAREAS_LEGIBLES[n]}: F1 por clase")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(salida / f"f1_por_clase_{n}.png", dpi=150)
        plt.close()


def _figuras_confusion(modelos, real, pred, salida):
    """Una figura por tarea, una matriz por modelo (normalizada por fila). El número es la proporción de la clase real."""
    modelos = list(modelos)
    for n in NOMBRES:
        K = K_CLASES[n]
        fig, ejes = plt.subplots(1, len(modelos), figsize=(3.6 * len(modelos) + 0.6, 3.8), squeeze=False)
        for ax, m in zip(ejes[0], modelos):
            M = matriz_confusion(real[n], pred[m][n], K)
            ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(K), CLASES[n], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(K), CLASES[n], fontsize=8)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.5 else "black")
            ax.set_title(MODELOS[m], fontsize=10)
            ax.set_xlabel("predicha")
            if ax is ejes[0][0]:
                ax.set_ylabel("real")
        fig.suptitle(f"{TAREAS_LEGIBLES[n]}: matriz de confusión (cada fila suma 1)", fontsize=11)
        plt.tight_layout()
        plt.savefig(salida / f"confusion_{n}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    evaluar()
