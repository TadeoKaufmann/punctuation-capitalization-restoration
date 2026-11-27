"""Protocolo común de todos los modelos (RF con y sin embeddings, LSTM, BiLSTM): mismos datos, mismo split y
misma métrica, para que la comparación sea justa.

Split: train / val / test por GRUPO, estratificado.
  - grupo = `documento` (la película, la conferencia o la nota de Wikinoticias), salvo en paenza y sintetico, que
    son un solo documento cada uno: ahí el grupo es el `bloque` (la oración). Así ninguna película, conferencia o
    nota aparece en dos conjuntos, y ninguna oración partida en pedazos queda a los dos lados.
  - estratificado (StratifiedGroupKFold) por las clases raras, para que la capitalización 2 y 3, el `¿` y el `?`
    no queden concentrados en un solo conjunto.
  - test ~20 %, val ~13 % del resto (1/7). Semillas fijas (ver SEMILLA_TEST).

Métricas: F1 macro por tarea a nivel palabra (la del enunciado), F1 por clase y matrices de confusión.

No depende de torch: lo usan también los scripts de Random Forest.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

NOMBRES = ["punt_inicial", "punt_final", "capitalizacion"]
K_CLASES = {"punt_inicial": 2, "punt_final": 4, "capitalizacion": 4}
MAPA = {
    "punt_inicial":   {"": 0, "¿": 1},
    "punt_final":     {"": 0, ",": 1, ".": 2, "?": 3},
    "capitalizacion": {0: 0, 1: 1, 2: 2, 3: 3},
}
# Nombre legible de cada clase (en el mismo orden que su id)
CLASES = {
    "punt_inicial":   ["nada", "¿"],
    "punt_final":     ["nada", ",", ".", "?"],
    "capitalizacion": ["0 minúscula", "1 Inicial", "2 mixta", "3 MAYÚS"],
}
FUENTES_SIN_DOCUMENTO = ("paenza", "sintetico")   # un solo documento cada una: se agrupa por bloque
# Semillas elegidas probando 40 de cada una y quedándose con la que mejor reparte (tamaño pedido y proporción de cada
# clase parecida a la del dataset entero). Solo se miró el reparto de las etiquetas, nunca el rendimiento de un modelo.
# Con pocos grupos grandes (34 películas) el reparto depende mucho de la semilla.
SEMILLA_TEST, SEMILLA_VAL = 13, 24
FOLDS_TEST, FOLDS_VAL = 5, 7                       # 1/5 = 20 % de test; 1/7 del resto = ~13 % de val


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_etiquetas(csv) -> pd.DataFrame:
    """etiquetas_palabras.csv con las tres etiquetas pasadas a ids (columnas `<tarea>_id`)."""
    et = pd.read_csv(csv, keep_default_na=False)
    for col, mapa in MAPA.items():
        et[col + "_id"] = et[col].map(mapa)
    assert not et[[c + "_id" for c in NOMBRES]].isna().any().any(), "hay etiquetas fuera del mapa"
    return et


def huella_datos(csv, X) -> str:
    """Identifica el dataset: hash de las etiquetas más la forma y una muestra de los embeddings. Es la misma cuenta
    que usa busqueda_lstm.Datos, así los resultados de todos los modelos quedan marcados con la misma huella."""
    return hashlib.md5(Path(csv).read_bytes() + str(X.shape).encode() + np.asarray(X[::5000]).tobytes()).hexdigest()[:8]


def asignar_grupo(et: pd.DataFrame) -> np.ndarray:
    """Un grupo por fila: `documento`, o el `bloque` en paenza y sintetico."""
    faltan = {"documento", "bloque", "fuente"} - set(et.columns)
    assert not faltan, (f"a etiquetas_palabras.csv le faltan las columnas {sorted(faltan)}: "
                        "regenerar los embeddings con generar_embeddings.ipynb")
    sin_doc = et["fuente"].isin(FUENTES_SIN_DOCUMENTO)
    return np.where(sin_doc, "bloque:" + et["bloque"].astype(str), "doc:" + et["documento"].astype(str))


def estratos(et: pd.DataFrame) -> np.ndarray:
    """Una categoría por palabra para estratificar, con prioridad a lo más escaso:
    4 capitalización 2, 3 capitalización 3, 2 `¿` o `?`, 1 coma o punto, 0 el resto."""
    e = np.zeros(len(et), dtype=int)
    e[et["punt_final"].isin([",", "."]).to_numpy()] = 1
    e[((et["punt_inicial"] == "¿") | (et["punt_final"] == "?")).to_numpy()] = 2
    e[(et["capitalizacion"] == 3).to_numpy()] = 3
    e[(et["capitalizacion"] == 2).to_numpy()] = 4
    return e


def dividir(et: pd.DataFrame):
    """(idx_train, idx_val, idx_test): posiciones de fila (las mismas de X y de etiquetas)."""
    grupo, estrato = asignar_grupo(et), estratos(et)
    cv_test = StratifiedGroupKFold(n_splits=FOLDS_TEST, shuffle=True, random_state=SEMILLA_TEST)
    idx_tv, idx_test = next(cv_test.split(et, estrato, grupo))
    cv_val = StratifiedGroupKFold(n_splits=FOLDS_VAL, shuffle=True, random_state=SEMILLA_VAL)
    i_tr, i_val = next(cv_val.split(idx_tv, estrato[idx_tv], grupo[idx_tv]))
    idx_train, idx_val = np.sort(idx_tv[i_tr]), np.sort(idx_tv[i_val])
    idx_test = np.sort(idx_test)
    g = lambda idx: set(grupo[idx])
    assert not (g(idx_train) & g(idx_val)) and not (g(idx_train) & g(idx_test)) and not (g(idx_val) & g(idx_test)), \
        "un grupo (documento o bloque) está en dos conjuntos"
    assert len(idx_train) + len(idx_val) + len(idx_test) == len(et)
    return idx_train, idx_val, idx_test


def huella_split(idx_train, idx_val, idx_test) -> str:
    """Identifica el split: cambia si cambia el dataset o la forma de partir."""
    h = hashlib.md5()
    for idx in (idx_train, idx_val, idx_test):
        h.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes()[::97])
        h.update(str(len(idx)).encode())
    return h.hexdigest()[:8]


def reporte_split(et: pd.DataFrame, idx_train, idx_val, idx_test, minimo=100) -> str:
    """Tabla de palabras por clase y por fuente en cada conjunto. Avisa si una clase queda con < `minimo` palabras."""
    grupo = asignar_grupo(et)
    partes = {"train": idx_train, "val": idx_val, "test": idx_test}
    total = len(et)
    lineas = ["Palabras por conjunto: " + " | ".join(f"{n} {len(i):,} ({100 * len(i) / total:.1f} %)" for n, i in partes.items()),
              "Grupos (documentos o bloques): " + " | ".join(f"{n} {len(set(grupo[i])):,}" for n, i in partes.items()), ""]
    avisos = []
    for tarea in NOMBRES:
        filas = []
        for k, nombre in enumerate(CLASES[tarea]):
            fila = {"clase": f"{tarea}: {nombre}"}
            for n, i in partes.items():
                c = int((et[tarea + "_id"].to_numpy()[i] == k).sum())
                fila[n] = f"{c:,} ({100 * c / len(i):.1f} %)"
                if c < minimo:
                    avisos.append(f"{tarea} '{nombre}' tiene solo {c} palabras en {n}")
            filas.append(fila)
        lineas.append(pd.DataFrame(filas).to_string(index=False))
        lineas.append("")
    por_fuente = pd.DataFrame({n: pd.Series(et["fuente"].to_numpy()[i]).value_counts() for n, i in partes.items()}).fillna(0).astype(int)
    lineas.append("Palabras por fuente:\n" + por_fuente.to_string())
    if avisos:
        lineas += ["", "AVISOS:"] + ["  - " + a for a in avisos]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def f1_macro(y, pred, K) -> float:
    return float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0))


def f1_por_clase(y, pred, K) -> np.ndarray:
    return f1_score(y, pred, labels=list(range(K)), average=None, zero_division=0)


def buscar_factores(P, y, K, pasadas=2, grilla=np.exp(np.linspace(-3, 3, 31))):
    """Factores w (w[0]=1) que maximizan el F1 macro de argmax(P * w), por búsqueda por coordenadas.
    Se ajustan con validación y se aplican tal cual en test."""
    w = np.ones(K)
    mejor = f1_macro(y, (P * w).argmax(1), K)
    for _ in range(pasadas):
        for k in range(1, K):
            for f in grilla:
                w2 = w.copy()
                w2[k] = f
                s = f1_macro(y, (P * w2).argmax(1), K)
                if s > mejor + 1e-9:
                    mejor, w = s, w2
    return w


def matriz_confusion(y, pred, K, normalizar=True) -> np.ndarray:
    """Filas = clase real, columnas = predicha. Normalizada por fila: cada fila suma 1 (el recall de cada clase)."""
    m = confusion_matrix(y, pred, labels=list(range(K))).astype(float)
    if normalizar:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    return m


# ---------------------------------------------------------------------------
# Predicciones en test (las guarda cada modelo; las lee src/evaluacion.py)
# ---------------------------------------------------------------------------
def guardar_predicciones(carpeta, nombre_modelo, filas, preds: dict, huella_datos: str, extra: dict | None = None):
    """Guarda las predicciones de test de un modelo: la fila global de cada palabra (posición en X/etiquetas) y la
    clase predicha (id) de cada tarea. `evaluacion.py` las compara con las etiquetas reales."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"fila": np.asarray(filas), **{"pred_" + n: np.asarray(preds[n]) for n in NOMBRES}})
    df = df.sort_values("fila").reset_index(drop=True)
    assert df["fila"].is_unique, "una palabra de test está repetida en las predicciones"
    df.to_csv(carpeta / "preds_test.csv", index=False)
    (carpeta / "preds_test.json").write_text(json.dumps(
        {"modelo": nombre_modelo, "dataset": huella_datos, "n_palabras": int(len(df)), **(extra or {})},
        indent=1, ensure_ascii=False), encoding="utf-8")
    return carpeta / "preds_test.csv"
