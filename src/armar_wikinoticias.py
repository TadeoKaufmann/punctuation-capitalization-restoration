"""Convierte las notas de Wikinoticias (datasets/wikinoticias/raw/notas.jsonl) al formato de instancias_final.

Salida: datasets/wikinoticias/instancias_wikinoticias.csv, con las MISMAS columnas que instancias_final.csv
(fuente = "wikinoticias", documento = titulo de la nota, Indice desde 400001). No toca instancias_final.

Motivacion: la capitalizacion clase 2 (mixta, tipo UPyD) y clase 3 (siglas, tipo ONU/OTAN) tienen muy pocos
ejemplos en el corpus (286 y 876 palabras). Wikinoticias aporta nombres de paises, organizaciones y siglas.

Limpieza: se descartan oraciones con signos que el TP no predice (: ; ! ... y cualquier caracter fuera de
letras, digitos, espacio y , . ? ¿ % - /), sin mayuscula inicial o sin cierre, repetidas, o con muchos numeros.
Se sacan comillas y parentesis (como en las conferencias); "EE. UU." se escribe EEUU.

Uso:  python src/armar_wikinoticias.py [--seed 42] [--max-notas N]
"""
import argparse
import json
import random
import re
from collections import Counter

import pandas as pd

from armar_instancias import clase_capitalizacion
from armar_instancias_final import CLASES, armar_grupo, cortar_por_largo
from rutas import DATASETS
from scrapear_conferencias import COMILLAS, MAX_DIGITOS_PALABRA, RE_PARENTESIS, RE_PARENTESIS_SIN_CERRAR, RE_PUNT_AJENA, dividir_oraciones

RAW = DATASETS / "wikinoticias" / "raw" / "notas.jsonl"
SALIDA = DATASETS / "wikinoticias" / "instancias_wikinoticias.csv"
BASE = 400001
MAX_FRAC_NUMERICAS = 0.2
RE_CARACTERES = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 ,.¿?%\-/]+$")
COMILLAS_SIN_APOSTROFE = str.maketrans("", "", "\"“”«»‘’")
RE_RUIDO_PARRAFO =re.compile(r"[|{}=<>]|https?://|www\.")


def limpiar_parrafo(p: str) -> str | None:
    if RE_RUIDO_PARRAFO.search(p):
        return None                                   # tablas, restos de plantillas, enlaces
    p = p.replace("&nbsp;", " ").replace("\xa0", " ")
    p = re.sub(r"EE\.?\s*UU\.?", "EEUU", p)
    p = RE_PARENTESIS.sub("", p)
    p = RE_PARENTESIS_SIN_CERRAR.sub("", p)
    p = re.sub(r"(?<=\w)[’'](?=\w)", "'", p)           # apostrofe dentro de una palabra (L'Aquila): se deja
    p = p.translate(COMILLAS_SIN_APOSTROFE)            # y la oracion se descarta ("caracter raro"); sacarlo crearia LAquila
    p = re.sub(r"\s+([,.?])", r"\1", p)
    return " ".join(p.split())


def motivo_invalida(o: str) -> str | None:
    """Por que se descarta la oracion, o None si sirve."""
    if not o.endswith((".", "?")):
        return "sin cierre"
    if not (o[0].isupper() or o[0] == "¿"):
        return "no empieza en mayuscula"
    if RE_PUNT_AJENA.search(o):
        return "signo ajeno"
    if not RE_CARACTERES.match(o):
        return "caracter raro"
    if o.count("¿") != o.count("?"):
        return "pregunta desbalanceada"
    palabras = o.split()
    if any(sum(c.isdigit() for c in w) > MAX_DIGITOS_PALABRA for w in palabras):
        return "numero largo"
    if sum(any(c.isdigit() for c in w) for w in palabras) / len(palabras) > MAX_FRAC_NUMERICAS:
        return "muy numerica"
    return None


def leer_notas(max_notas: int | None, descartadas: Counter) -> dict:
    vistas: set = set()

    def invalida(o: str) -> str | None:
        motivo = motivo_invalida(o)
        if motivo is None:
            if o in vistas:
                return "repetida"
            vistas.add(o)
        return motivo

    grupos = {}
    with RAW.open(encoding="utf-8") as f:
        for linea in f:
            nota = json.loads(linea)
            oraciones = []
            for parrafo in nota["texto"].split("\n"):
                p = limpiar_parrafo(parrafo)
                if p:
                    oraciones += [(o, f"wikinoticias:{nota['pageid']}") for o in dividir_oraciones(p)]
            tramos = cortar_por_largo(oraciones, descartadas, "wikinoticias", invalida)
            if tramos:
                grupos[nota["titulo"].strip()] = tramos
            if max_notas and len(grupos) >= max_notas:
                break
    return grupos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-notas", type=int, default=None)
    args = ap.parse_args()

    descartadas: Counter = Counter()
    grupos = leer_notas(args.max_notas, descartadas)
    filas, n = [], 0
    for titulo, tramos in grupos.items():
        rng = random.Random(f"{args.seed}-wikinoticias-{titulo}")
        for j, inst in enumerate(armar_grupo(tramos, rng, descartadas, "wikinoticias")):
            primera = inst.pop("_primera")
            filas.append({"Indice": BASE + n + j, "documento": titulo, "fuente": "wikinoticias",
                          "bloque": BASE + n + primera, **inst})
        n = len(filas)
    df = pd.DataFrame(filas)
    columnas = ["Indice", "Fragmento", "documento", "fuente", "clase", "parte", "n_oraciones", "n_palabras", "origen", "bloque"]
    df = df[columnas]
    assert df["Indice"].is_unique and (df["Fragmento"].str.len() > 0).all()
    SALIDA.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))

    print(f"{len(grupos)} notas, {len(df)} instancias, {int(df['n_palabras'].sum())} palabras -> {SALIDA}")
    print(df["clase"].value_counts().reindex(CLASES).to_string())
    caps = Counter(clase_capitalizacion(w.strip(".,?¿%-/")) for t in df["Fragmento"] for w in t.split() if w.strip(".,?¿%-/"))
    print("capitalizacion por palabra:", dict(sorted(caps.items())))
    print("descartes:", {k: v for k, v in descartadas.items() if k.startswith("wikinoticias:") and "corte" not in k and "achic" not in k})


if __name__ == "__main__":
    main()
