"""
Datos para el Random Forest: una fila por PALABRA de cada instancia (fragmento),
con su embedding = promedio de los embeddings de los tokens BERT que la forman
(mismo procedimiento que Clase_P7.ipynb, sección 2.2).

Estructura: funciones auxiliares arriba, main() al final que las orquesta.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")                    # sin ventana: solo guardamos la imagen
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from transformers import BertTokenizer, BertModel

# Reusamos la lectura/filtrado y la regex que separa puntuación de la palabra
from armar_instancias import (
    PALABRA_RE, INDICE_DESDE, filtrar_fragmentos, puntuacion_final, clase_capitalizacion,
)
from rutas import DATASETS, SALIDAS

MODELO_BERT = "bert-base-multilingual-cased"
# Se borran solo estos signos (los que predice el TP); tildes, espacios y el resto quedan igual
SIGNOS_A_BORRAR = str.maketrans("", "", ",.?¿")


# ---------------------------------------------------------------------------
# Auxiliares: texto
# ---------------------------------------------------------------------------
def cargar_fragmentos_filtrados(ruta_csv=DATASETS / "fragmentos_todos.csv", desde: int = INDICE_DESDE) -> pd.DataFrame:
    """Fragmentos (Indice, Fragmento) desde `desde`, sin cuentas ni texto editorial."""
    df = pd.read_csv(ruta_csv)
    df = df[df["Indice"] >= desde].dropna(subset=["Fragmento"])
    return filtrar_fragmentos(df)


def ruta_instancias_final() -> Path:
    """datasets/instancias_final/instancias_final.csv en el repo; en Colab (rutas.py: todo en una carpeta)
    el archivo está suelto junto a los .py."""
    for ruta in (DATASETS / "instancias_final" / "instancias_final.csv", DATASETS / "instancias_final.csv"):
        if ruta.exists():
            return ruta
    raise FileNotFoundError("No encuentro instancias_final.csv (ni en datasets/instancias_final/ ni junto a los .py)")


def cargar_instancias(ruta_csv=None, filtrar: bool = False) -> pd.DataFrame:
    """Instancias de entrenamiento (datasets/instancias_final/instancias_final.csv, generado por
    armar_instancias_final.py). Conserva todas las columnas: Indice, Fragmento, documento, fuente, clase, ...

    `filtrar=False` (por defecto): el CSV ya viene filtrado desde armar_instancias_final.py; filtrar de nuevo
    podría borrar instancias sueltas y dejar oraciones partidas incompletas.
    """
    df = pd.read_csv(ruta_csv or ruta_instancias_final()).dropna(subset=["Fragmento"])
    assert df["Indice"].is_unique, "Indice repetido en las instancias"
    return filtrar_fragmentos(df) if filtrar else df


def sha256_archivo(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def normalizar_texto(texto: str) -> str:
    """Minúsculas y sin comas, puntos ni signos de pregunta (¿ ?). Conserva tildes y espacios."""
    return texto.lower().translate(SIGNOS_A_BORRAR)


def palabras_y_etiquetas(texto: str) -> list[dict]:
    """Una fila por palabra del fragmento, en orden: la palabra limpia (entrada) y sus 3 etiquetas.

    - palabra:         minúsculas y sin puntuación pegada -> es lo que se convierte en embedding.
    - punt_inicial:    "¿" o "" (si la palabra abre una pregunta).
    - punt_final:      "," "." "?" o "" (primer signo que la sigue).
    - capitalizacion:  0 minúscula, 1 Inicial mayúscula, 2 mixta (McDonald's), 3 TODO MAYÚSCULA.
    Las etiquetas se sacan del texto ORIGINAL, antes de limpiarlo.
    """
    filas = []
    for palabra in texto.split():
        prefijo, nucleo, sufijo = PALABRA_RE.match(palabra).groups()
        if not nucleo:                   # "palabras" que eran solo puntuación (ej. "-")
            continue
        filas.append({
            "palabra": nucleo.lower(),
            "punt_inicial": "¿" if "¿" in prefijo else "",
            "punt_final": puntuacion_final(sufijo),
            "capitalizacion": clase_capitalizacion(nucleo),
        })
    return filas


def guardar_textos(df: pd.DataFrame, salida=SALIDAS / "instancias_textos.csv") -> None:
    """CSV con: instancia_id | texto_original (como en el libro) | texto_normalizado."""
    pd.DataFrame({
        "instancia_id": df["Indice"].astype(int),
        "texto_original": df["Fragmento"],
        "texto_normalizado": df["Fragmento"].map(normalizar_texto),
    }).to_csv(salida, index=False)       # pandas pone entre comillas los textos con ","


# ---------------------------------------------------------------------------
# Auxiliares: embeddings
# ---------------------------------------------------------------------------
def cargar_bert():
    """Devuelve (tokenizer, matriz): matriz es (vocabulario x 768), una fila por token.

    Es la capa de entrada de BERT: estática, el vector de un token no depende del contexto.
    """
    tokenizer = BertTokenizer.from_pretrained(MODELO_BERT)
    modelo = BertModel.from_pretrained(MODELO_BERT)
    matriz = modelo.get_input_embeddings().weight.detach().numpy()
    return tokenizer, matriz


def embedding_palabra(palabra: str, tokenizer, matriz: np.ndarray) -> np.ndarray | None:
    """Embedding de una palabra = promedio de los vectores de sus tokens (vector de 768)."""
    # "cuándo" -> ['cu', '##ánd', '##o'] -> sus 3 ids (lista de enteros = filas de `matriz`)
    ids = tokenizer(palabra, add_special_tokens=False)["input_ids"]
    if not ids:                          # sin tokens el promedio daría NaN
        return None
    # matriz[ids] es (len(ids), 768); axis=0 promedia las filas, columna por columna
    return matriz[ids].mean(axis=0)


# ---------------------------------------------------------------------------
# Auxiliares: armado de X e y
# ---------------------------------------------------------------------------
def armar_X_y(instancias: dict[int, list[dict]], tokenizer, matriz: np.ndarray):
    """Apila todas las palabras de todas las instancias, en orden.

    Devuelve (X, etiquetas):
      X         (n_palabras_total, 768) float32; la fila i es el embedding de la palabra i.
      etiquetas DataFrame con una fila por palabra (misma fila i que X): instancia_id, posicion,
                palabra, punt_inicial, punt_final, capitalizacion.
    """
    cache = {}                           # palabra -> embedding; "de", "que"... se repiten miles de veces
    bloques, filas_etiquetas = [], []
    for iid, filas in instancias.items():
        for pos, f in enumerate(filas):
            palabra = f["palabra"]
            if palabra not in cache:
                emb = embedding_palabra(palabra, tokenizer, matriz)
                # Sin tokens -> ceros, así no se corre el alineamiento con las etiquetas
                cache[palabra] = emb if emb is not None else np.zeros(matriz.shape[1])
            bloques.append(cache[palabra])
            filas_etiquetas.append({"instancia_id": iid, "posicion": pos, **f})
    X = np.vstack(bloques).astype(np.float32)
    return X, pd.DataFrame(filas_etiquetas)


def agregar_metadatos(etiquetas: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a cada palabra de `etiquetas` la `fuente`, la `clase` de su instancia (parte, 1_oracion, ...), su
    `bloque` y su `documento` (la película, conferencia o nota de donde sale), para poder analizar por fuente o
    clase y para partir train/val/test por documento (src/protocolo.py). Van al final: las 3 etiquetas y
    `instancia_id` no se mueven. Las columnas que `df` no trae (CSV viejo) simplemente no se agregan."""
    cols = {"fuente": "fuente", "clase": "clase_instancia", "bloque": "bloque", "documento": "documento"}
    por_id = df.set_index("Indice")
    for origen, destino in cols.items():
        if origen in por_id.columns:
            etiquetas[destino] = etiquetas["instancia_id"].map(por_id[origen])
    return etiquetas


def verificar_run(X: np.ndarray, etiquetas: pd.DataFrame, df: pd.DataFrame) -> None:
    """Chequeos antes de guardar: X y etiquetas alineados, sin NaN, y todas las instancias del CSV presentes."""
    assert len(X) == len(etiquetas), "X y etiquetas no tienen las mismas filas"
    assert not np.isnan(X).any(), "hay NaN en X"
    faltan = set(df["Indice"]) - set(etiquetas["instancia_id"])
    assert not faltan, f"instancias sin ninguna palabra: {sorted(faltan)[:5]}... ({len(faltan)})"


def guardar_run(X, etiquetas, ruta_instancias, carpeta=SALIDAS) -> None:
    """Guarda el run completo: embeddings_palabras.npy + etiquetas_palabras.csv (misma fila i) y
    embeddings_meta.json con el sha256 del CSV de instancias que los generó, para saber si quedaron viejos."""
    carpeta = Path(carpeta)
    np.save(carpeta / "embeddings_palabras.npy", X)
    etiquetas.to_csv(carpeta / "etiquetas_palabras.csv", index=False)
    meta = {"instancias_csv": Path(ruta_instancias).name, "sha256_instancias": sha256_archivo(ruta_instancias),
            "n_instancias": int(etiquetas["instancia_id"].nunique()), "n_palabras": int(len(etiquetas)),
            "generado": time.strftime("%Y-%m-%d %H:%M:%S")}
    (carpeta / "embeddings_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("guardado en", carpeta, meta)


def dividir_por_instancia(etiquetas: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Devuelve (idx_train, idx_test): posiciones de fila (las mismas de X y de etiquetas).

    Se reparte por instancia_id, no por palabra: todas las palabras de un fragmento caen del
    mismo lado. Si no, palabras vecinas estarían en train y test y la evaluación saldría inflada.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_test = next(splitter.split(etiquetas, groups=etiquetas["instancia_id"]))
    return idx_train, idx_test


ETIQUETAS = ["punt_inicial", "punt_final", "capitalizacion"]


def entrenar_y_evaluar(X_train, X_test, y_train, y_test):
    """Un Random Forest por etiqueta. Devuelve {etiqueta: modelo}."""
    modelos = {}
    for col in ETIQUETAS:
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",   # da más peso a las clases raras (¿, ?, mayúsculas mixtas)
            n_jobs=-1,                 # usa todos los núcleos
            random_state=42,
        )
        rf.fit(X_train, y_train[col])
        print(f"\n=== {col} ===", flush=True)
        # precision/recall por clase: la exactitud total engaña porque "nada" es ~90% de los datos
        print(classification_report(y_test[col], rf.predict(X_test), zero_division=0), flush=True)
        modelos[col] = rf
    return modelos


def graficar_matrices_confusion(modelos, X_test, y_test, salida=SALIDAS / "matrices_confusion.png"):
    """Una matriz de confusión por etiqueta: filas = clase real, columnas = clase predicha."""
    fig, ejes = plt.subplots(1, len(modelos), figsize=(6 * len(modelos), 5))
    for eje, (col, rf) in zip(ejes, modelos.items()):
        pred = rf.predict(X_test)
        clases = list(rf.classes_)
        cuentas = confusion_matrix(y_test[col], pred, labels=clases)
        # Cada fila dividida por su total -> el color y el % son el recall de cada clase real
        porcentajes = cuentas / cuentas.sum(axis=1, keepdims=True)
        eje.imshow(porcentajes, cmap="viridis", vmin=0, vmax=1)
        for i in range(len(clases)):
            for j in range(len(clases)):
                color = "black" if porcentajes[i, j] > 0.5 else "white"
                eje.text(j, i, f"{cuentas[i, j]}\n({porcentajes[i, j]:.0%})",
                         ha="center", va="center", color=color, fontsize=9)
        nombres = [repr(c) if c == "" else str(c) for c in clases]   # "nada" se ve como ''
        eje.set_xticks(range(len(clases)), nombres)
        eje.set_yticks(range(len(clases)), nombres)
        eje.set_xlabel("Predicho")
        eje.set_ylabel("Real")
        eje.set_title(col)
        # La misma matriz con cantidades absolutas, para ver cuántas palabras son
        print(f"\n=== {col} (cantidades; filas = real, columnas = predicho) ===")
        print(pd.crosstab(y_test[col].rename("real"), pd.Series(pred, index=y_test.index, name="pred")))
    fig.tight_layout()
    fig.savefig(salida, dpi=120)


# ---------------------------------------------------------------------------
def main(solo_embeddings: bool = False):
    ruta = ruta_instancias_final()
    df = cargar_instancias(ruta)
    guardar_textos(df)

    # {instancia_id: [{palabra, punt_inicial, punt_final, capitalizacion}, ...]}
    instancias = {int(i): palabras_y_etiquetas(t) for i, t in zip(df["Indice"], df["Fragmento"])}
    print(f"{len(instancias)} instancias, {sum(map(len, instancias.values()))} palabras en total")

    tokenizer, matriz = cargar_bert()
    X, etiquetas = armar_X_y(instancias, tokenizer, matriz)
    etiquetas = agregar_metadatos(etiquetas, df)
    print(f"X: {X.shape} | etiquetas: {etiquetas.shape}")

    # X pesa cientos de MB: va en .npy; el CSV liviano tiene las etiquetas (misma fila i)
    verificar_run(X, etiquetas, df)
    guardar_run(X, etiquetas, ruta)
    if solo_embeddings:                  # sin entrenar el Random Forest
        return

    idx_train, idx_test = dividir_por_instancia(etiquetas)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train = etiquetas.iloc[idx_train]      # tiene las 3 columnas de etiquetas
    y_test = etiquetas.iloc[idx_test]
    print(f"train: {len(idx_train)} palabras en {y_train['instancia_id'].nunique()} instancias | "
          f"test: {len(idx_test)} palabras en {y_test['instancia_id'].nunique()} instancias")

    modelos = entrenar_y_evaluar(X_train, X_test, y_train, y_test)
    graficar_matrices_confusion(modelos, X_test, y_test)


if __name__ == "__main__":
    import sys
    # python src/embeddings_palabras.py --solo-embeddings   -> genera y guarda en salidas/, sin Random Forest
    main(solo_embeddings="--solo-embeddings" in sys.argv)
