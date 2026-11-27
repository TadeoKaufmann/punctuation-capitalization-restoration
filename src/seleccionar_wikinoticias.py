"""Elige de instancias_wikinoticias.csv las instancias con mas palabras de capitalizacion clase 2 y 3.

Salida: datasets/wikinoticias/instancias_wikinoticias_sel.csv (mismas columnas). No se mezcla con instancias_final.

Criterio:
  - La unidad es el `bloque`: si se elige una instancia `parte`, entran TODOS los pedazos de esa oracion.
  - Solo entran bloques con al menos una palabra de clase 2 o 3.
  - Se priorizan los bloques con palabras de tipos RAROS (poco frecuentes en las notas), asi hay variedad
    de siglas y nombres en vez de 400 veces "ONU".
  - Tope por tipo de palabra (--tope-tipo): un tipo (ej. ONU) no aparece mas de N veces en la seleccion.
  - Se corta al llegar a --palabras (por defecto 125.000).

Uso:  python src/seleccionar_wikinoticias.py [--palabras 125000] [--tope-tipo 30]
"""
import argparse
from collections import Counter

import pandas as pd

from armar_instancias import clase_capitalizacion
from rutas import DATASETS

ENTRADA = DATASETS / "wikinoticias" / "instancias_wikinoticias.csv"
SALIDA = DATASETS / "wikinoticias" / "instancias_wikinoticias_sel.csv"
BORDES = ".,?¿%-/"


def palabras_cap(texto: str) -> list[tuple[str, int]]:
    """[(palabra, clase)] de las palabras de clase 2 o 3 del texto."""
    out = []
    for w in texto.split():
        w = w.strip(BORDES)
        if w:
            c = clase_capitalizacion(w)
            if c in (2, 3):
                out.append((w, c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palabras", type=int, default=125_000)
    ap.add_argument("--tope-tipo", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(ENTRADA, keep_default_na=False)
    df["_cap"] = df["Fragmento"].map(palabras_cap)
    frecuencia = Counter(w for caps in df["_cap"] for w, _ in caps)

    bloques = []
    for bloque, g in df.groupby("bloque", sort=False):
        caps = [p for lista in g["_cap"] for p in lista]
        if caps:
            # los tipos raros pesan mas; la clase 2 (la mas escasa) pesa el doble
            puntaje = sum((2 if c == 2 else 1) / frecuencia[w] for w, c in caps)
            bloques.append((puntaje, bloque, caps, int(g["n_palabras"].sum())))
    bloques.sort(key=lambda b: (-b[0], b[1]))

    elegidos, usados, total = [], Counter(), 0
    for puntaje, bloque, caps, n in bloques:
        if total >= args.palabras:
            break
        if all(usados[w] >= args.tope_tipo for w, _ in caps):
            continue                                   # todo lo que aporta ya esta de sobra
        elegidos.append(bloque)
        usados.update(w for w, _ in caps)
        total += n

    sel = df[df["bloque"].isin(set(elegidos))].drop(columns="_cap").sort_values("Indice")
    SALIDA.write_bytes(sel.to_csv(index=False, lineterminator="\n").encode("utf-8"))

    caps = Counter(c for t in sel["Fragmento"] for _, c in palabras_cap(t))
    tipos = {c: len({w for t in sel["Fragmento"] for w, k in palabras_cap(t) if k == c}) for c in (2, 3)}
    print(f"{len(sel)} instancias, {int(sel['n_palabras'].sum())} palabras, {sel['documento'].nunique()} notas -> {SALIDA}")
    print(sel["clase"].value_counts().to_string())
    print(f"palabras clase 2: {caps[2]} ({tipos[2]} tipos) | clase 3: {caps[3]} ({tipos[3]} tipos)")
    print("mas repetidas:", Counter({w: n for w, n in usados.items()}).most_common(12))


if __name__ == "__main__":
    main()
