"""
Inferencia: lee un CSV de tokens SIN etiquetas (el formato de datos_test.csv de la cátedra) y escribe el CSV de
predicciones que pide el enunciado: una línea por token con punt_inicial, punt_final y capitalización.

    instancia_id,token_id,token,punt_inicial,punt_final,capitalización

Cómo pasa de tokens a palabras (la evaluación es por palabra): un token que empieza con `##` continúa la palabra del
token anterior; el resto abre una palabra nueva. El embedding de la palabra es el promedio de los vectores de entrada
de BERT de sus tokens (igual que en el entrenamiento, src/embeddings_palabras.py). Después se predice por palabra y
se copia al CSV de tokens así:
    punt_inicial -> primer token de la palabra
    punt_final   -> último token de la palabra
    capitalización -> todos los tokens de la palabra
Los factores por clase (umbrales) que se ajustaron con validación se aplican igual que en la evaluación.

Uso (desde la raíz del repo; el modelo es el que dejó el entrenamiento):
    python src/inferir.py --modelo bilstm   --entrada datos_test.csv --salida predicciones.csv
    python src/inferir.py --modelo lstm     --entrada datos_test.csv --salida predicciones.csv
    python src/inferir.py --modelo rf_sin_emb --entrada datos_test.csv --salida predicciones.csv
    python src/inferir.py --modelo rf_con_emb --entrada datos_test.csv --salida predicciones.csv
    (--modelo-ruta para usar un archivo de modelo en otro lugar; --rapido usa las carpetas *_rapido)

Requiere:  salidas/<modelo>/modelo_final.pt (redes) o modelo_rf.joblib (RF, entrenado con --guardar-modelo),
y la librería transformers para leer la matriz de embeddings de bert-base-multilingual-cased.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "entrenamiento"))   # carpeta de entrega: código compartido (rutas, protocolo, ...)
from rutas import SALIDAS  # noqa: E402

NOMBRES = ["punt_inicial", "punt_final", "capitalizacion"]
K_CLASES = {"punt_inicial": 2, "punt_final": 4, "capitalizacion": 4}
SIMBOLO_INICIAL = {0: "", 1: "¿"}
SIMBOLO_FINAL = {0: "", 1: ",", 2: ".", 3: "?"}
MODELO_BERT = "bert-base-multilingual-cased"
COLUMNAS_SALIDA = ["instancia_id", "token_id", "token", "punt_inicial", "punt_final", "capitalización"]


# ---------------------------------------------------------------------------
# Entrada: tokens -> palabras
# ---------------------------------------------------------------------------
def leer_tokens(ruta) -> pd.DataFrame:
    df = pd.read_csv(ruta, keep_default_na=False)
    faltan = {"instancia_id", "token_id", "token"} - set(df.columns)
    assert not faltan, f"al CSV de entrada le faltan columnas: {sorted(faltan)}"
    df["token_id"] = df["token_id"].astype(int)
    return df


def agrupar_palabras(df: pd.DataFrame) -> np.ndarray:
    """Índice de palabra (0, 1, 2, ...) de cada token, global. Una palabra nunca cruza instancias."""
    es_continuacion = df["token"].str.startswith("##").to_numpy()
    nueva_instancia = np.r_[True, df["instancia_id"].to_numpy()[1:] != df["instancia_id"].to_numpy()[:-1]]
    return np.cumsum(~es_continuacion | nueva_instancia) - 1


def embeddings_de_palabras(df: pd.DataFrame, palabra_de_token: np.ndarray) -> np.ndarray:
    """(n_palabras, 768): promedio de los embeddings estáticos de los tokens de cada palabra."""
    from transformers import BertModel
    matriz = BertModel.from_pretrained(MODELO_BERT).get_input_embeddings().weight.detach().numpy()
    ids = df["token_id"].to_numpy()
    assert ids.max() < len(matriz), "hay token_id fuera del vocabulario de BERT"
    n = int(palabra_de_token.max()) + 1
    suma = np.zeros((n, matriz.shape[1]), dtype=np.float64)
    np.add.at(suma, palabra_de_token, matriz[ids])
    cuenta = np.bincount(palabra_de_token, minlength=n)[:, None]
    return (suma / cuenta).astype(np.float32)


def textos_de_palabras(df: pd.DataFrame, palabra_de_token: np.ndarray) -> list:
    """El texto en minúsculas de cada palabra (los tokens sin `##`), para los atributos del RF sin embeddings."""
    piezas = df["token"].str.replace("##", "", regex=False).str.lower().to_numpy()
    textos = [""] * (int(palabra_de_token.max()) + 1)
    for w, p in zip(palabra_de_token, piezas):
        textos[w] += p
    return textos


# ---------------------------------------------------------------------------
# Modelos: devuelven las probabilidades (n_palabras, K) de cada tarea
# ---------------------------------------------------------------------------
def probs_red(ruta, embeddings, instancia_de_palabra):
    import torch
    import busqueda_lstm as bl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    modelo = bl.ARQUITECTURAS[cfg.get("arq", "lstm")](H=cfg["H"], L=cfg["L"], dropout=cfg["dropout"]).to(device)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    print(f"red {cfg.get('arq', 'lstm')} (H={cfg['H']}, L={cfg['L']}) | dataset de entrenamiento {ck.get('dataset')} | "
          f"semilla {ck.get('seed')} | device {device}")

    # una secuencia por instancia; se procesan de a lotes de largo parecido
    inicios = np.flatnonzero(np.r_[True, instancia_de_palabra[1:] != instancia_de_palabra[:-1]])
    fines = np.r_[inicios[1:], len(instancia_de_palabra)]
    orden = np.argsort(fines - inicios, kind="stable")
    P = [np.zeros((len(embeddings), K_CLASES[n]), dtype=np.float32) for n in NOMBRES]
    with torch.no_grad():
        for j in range(0, len(orden), 32):
            lote = orden[j:j + 32]
            xs = [torch.from_numpy(embeddings[inicios[i]:fines[i]]) for i in lote]
            x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
            logits = modelo(x)
            for k in range(3):
                pr = torch.softmax(logits[k], dim=-1).cpu().numpy()
                for b, i in enumerate(lote):
                    P[k][inicios[i]:fines[i]] = pr[b, :fines[i] - inicios[i]]
    factores = ck["factores"] if ck.get("usa_umbral") else [np.ones(K_CLASES[n]) for n in NOMBRES]
    return P, [np.asarray(f) for f in factores]


def probs_rf(ruta, embeddings, df, palabra_de_token, instancia_de_palabra):
    import joblib
    from protocolo import NOMBRES as N
    guardado = joblib.load(ruta)
    assert "bosques" in guardado, ("este modelo_rf.joblib es de una versión vieja: volver a correr "
                                   "busqueda_rf.py con --guardar-modelo")
    print(f"random forest {guardado['modelo']} | dataset de entrenamiento {guardado['dataset']}")
    if guardado["modelo"] == "con_emb":
        X = embeddings
    else:
        import atributos_rf
        textos = textos_de_palabras(df, palabra_de_token)
        X = atributos_rf.construir_atributos(
            pd.DataFrame({"instancia_id": instancia_de_palabra, "_norm": textos}), guardado["vocab"])
    P, factores = [], []
    for n in N:
        rf = guardado["bosques"][n]
        p = np.zeros((X.shape[0], K_CLASES[n]), dtype=np.float32)
        p[:, rf.classes_] = rf.predict_proba(X)
        P.append(p)
        factores.append(np.asarray(guardado["factores"][n]))
    return P, factores


# ---------------------------------------------------------------------------
# Salida: palabras -> tokens
# ---------------------------------------------------------------------------
def armar_salida(df, palabra_de_token, clases):
    """clases: {tarea: (n_palabras,) ids}. Devuelve el DataFrame en el formato del enunciado, mismo orden que la entrada."""
    es_primero = np.r_[True, palabra_de_token[1:] != palabra_de_token[:-1]]
    es_ultimo = np.r_[palabra_de_token[1:] != palabra_de_token[:-1], True]
    ini = np.array([SIMBOLO_INICIAL[c] for c in clases["punt_inicial"][palabra_de_token]], dtype=object)
    fin = np.array([SIMBOLO_FINAL[c] for c in clases["punt_final"][palabra_de_token]], dtype=object)
    ini[~es_primero] = ""
    fin[~es_ultimo] = ""
    out = df[["instancia_id", "token_id", "token"]].copy()
    out["punt_inicial"] = ini
    out["punt_final"] = fin
    out["capitalización"] = clases["capitalizacion"][palabra_de_token]
    return out[COLUMNAS_SALIDA]


def predecir_tokens(df, modelo, ruta):
    """Del DataFrame de tokens (leer_tokens) al DataFrame de predicciones en el formato del enunciado."""
    palabra_de_token = agrupar_palabras(df)
    n_palabras = int(palabra_de_token.max()) + 1
    # instancia de cada palabra (para armar las secuencias); las filas de una instancia son contiguas
    instancia_de_palabra = df["instancia_id"].to_numpy()[np.r_[True, palabra_de_token[1:] != palabra_de_token[:-1]]]
    print(f"{len(df):,} tokens | {n_palabras:,} palabras | {df['instancia_id'].nunique():,} instancias")
    es_red = modelo in ("lstm", "bilstm")
    emb = embeddings_de_palabras(df, palabra_de_token) if (es_red or modelo == "rf_con_emb") else None
    if es_red:
        P, factores = probs_red(ruta, emb, instancia_de_palabra)
    else:
        P, factores = probs_rf(ruta, emb, df, palabra_de_token, instancia_de_palabra)
    clases = {n: (P[i] * factores[i]).argmax(1) for i, n in enumerate(NOMBRES)}
    return armar_salida(df, palabra_de_token, clases)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, choices=["lstm", "bilstm", "rf_sin_emb", "rf_con_emb"])
    ap.add_argument("--entrada", required=True, help="CSV de tokens sin etiquetas (como datos_test.csv)")
    ap.add_argument("--salida", default="predicciones.csv")
    ap.add_argument("--modelo-ruta", default=None, help="archivo del modelo (por defecto, el de salidas/<modelo>/)")
    ap.add_argument("--rapido", action="store_true", help="usa las carpetas *_rapido (prueba de humo)")
    a = ap.parse_args()

    df = leer_tokens(a.entrada)
    carpeta = SALIDAS / (a.modelo + ("_rapido" if a.rapido else ""))
    es_red = a.modelo in ("lstm", "bilstm")
    ruta = Path(a.modelo_ruta) if a.modelo_ruta else carpeta / ("modelo_final.pt" if es_red else "modelo_rf.joblib")
    assert ruta.exists(), f"no encuentro el modelo en {ruta}" + ("" if es_red else " (correr busqueda_rf.py con --guardar-modelo)")
    out = predecir_tokens(df, a.modelo, ruta)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.salida, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")   # la coma queda entre comillas
    print(f"predicciones guardadas en {a.salida} ({len(out):,} filas)")
    for n, col in [("punt_inicial", "punt_inicial"), ("punt_final", "punt_final"), ("capitalizacion", "capitalización")]:
        print(f"  {col}: " + ", ".join(f"{k!r}: {v:,}" for k, v in out[col].value_counts().items()))


if __name__ == "__main__":
    main()
