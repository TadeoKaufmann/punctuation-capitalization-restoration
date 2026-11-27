"""Distribución de las etiquetas (¿ , . ? y capitalización) por fuente y por película.

Usa el mismo etiquetado y los mismos filtros que el pipeline (embeddings_palabras.palabras_y_etiquetas y
cargar_fragmentos_filtrados), así que los números son los de las palabras que realmente entran a la red.
No necesita transformers (se simula en memoria: el etiquetado no usa BERT).

    python src/distribucion_etiquetas.py            # imprime tablas y guarda CSV en salidas/distribucion/
"""
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("transformers")

    class _Falso:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return None

    stub.BertTokenizer = stub.BertModel = _Falso
    sys.modules["transformers"] = stub

from embeddings_palabras import cargar_fragmentos_filtrados, palabras_y_etiquetas  # noqa: E402
from rutas import DATASETS, SALIDAS  # noqa: E402

SALIDA = SALIDAS / "distribucion"


def filas_de_fragmentos(df: pd.DataFrame, fuente_de) -> list[dict]:
    """Una fila por palabra, con fuente, grupo (película / conferencia), instancia y posición."""
    filas = []
    for indice, texto, grupo in zip(df["Indice"], df["Fragmento"], df["Conferencia"]):
        for pos, w in enumerate(palabras_y_etiquetas(texto)):
            filas.append({"fuente": fuente_de(indice), "grupo": grupo, "instancia": int(indice), "pos": pos, **w})
    return filas


def filas_sinteticas() -> list[dict]:
    filas = []
    for linea in (DATASETS / "dataset_sintetico" / "dataset_sintetico.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(linea)
        for pos, (p, i, f, c) in enumerate(zip(r["palabras"], r["punt_ini"], r["punt_fin"], r["cap"])):
            filas.append({"fuente": "sintetico", "grupo": r["categoria"], "instancia": r["id"], "pos": pos,
                          "palabra": p, "punt_inicial": i, "punt_final": f, "capitalizacion": c})
    return filas


def preguntas(w: pd.DataFrame) -> pd.DataFrame:
    """Una fila por pregunta: abre con ¿ y cierra en el primer ? (o queda sin cerrar dentro del fragmento).
    También cuenta los ? que cierran sin ¿ (cierres huérfanos)."""
    out = []
    for (fuente, grupo, inst), g in w.groupby(["fuente", "grupo", "instancia"], sort=False):
        ini, fin = g["punt_inicial"].tolist(), g["punt_final"].tolist()
        usados = set()
        for i, p in enumerate(ini):
            if p != "¿":
                continue
            j = next((k for k in range(i, len(fin)) if fin[k] == "?"), None)
            if j is not None:
                usados.add(j)
            previo = "inicio" if i == 0 else {".": "tras punto", "?": "tras punto", ",": "tras coma"}.get(fin[i - 1], "tras palabra sin signo")
            out.append({"fuente": fuente, "grupo": grupo, "instancia": inst, "largo": (j - i + 1) if j is not None else None,
                        "abre": previo, "cerrada": j is not None})
        huerfanos = [k for k, f in enumerate(fin) if f == "?" and k not in usados]
        for k in huerfanos:
            out.append({"fuente": fuente, "grupo": grupo, "instancia": inst, "largo": None, "abre": "huérfano (? sin ¿)", "cerrada": True})
    return pd.DataFrame(out)


def tabla(w: pd.DataFrame, por) -> pd.DataFrame:
    n = w.groupby(por).size().rename("palabras")
    t = pd.DataFrame({"palabras": n, "instancias": w.groupby(por)["instancia"].nunique()})
    pct = lambda mask: (100 * mask.groupby([w[c] for c in por]).mean()).round(2)
    t["¿ %"] = pct(w["punt_inicial"] == "¿")
    t["? %"] = pct(w["punt_final"] == "?")
    t["coma %"] = pct(w["punt_final"] == ",")
    t["punto %"] = pct(w["punt_final"] == ".")
    t["cap1 %"] = pct(w["capitalizacion"] == 1)
    t["cap2 %"] = pct(w["capitalizacion"] == 2)
    t["cap3 %"] = pct(w["capitalizacion"] == 3)
    t["n ¿"] = (w["punt_inicial"] == "¿").groupby([w[c] for c in por]).sum()
    t["n ?"] = (w["punt_final"] == "?").groupby([w[c] for c in por]).sum()
    t["n cap2"] = (w["capitalizacion"] == 2).groupby([w[c] for c in por]).sum()
    t["n cap3"] = (w["capitalizacion"] == 3).groupby([w[c] for c in por]).sum()
    con_preg = w.assign(q=(w["punt_inicial"] == "¿") | (w["punt_final"] == "?")).groupby(por + ["instancia"])["q"].any()
    t["% instancias con pregunta"] = (100 * con_preg.groupby(por).mean()).round(1)
    return t


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar_fragmentos_filtrados()                       # fragmentos_todos.csv ya con los filtros del pipeline
    fuente_de = lambda i: "paenza" if i < 100000 else ("conferencias" if i < 200000 else "subtitulos")
    w = pd.DataFrame(filas_de_fragmentos(df, fuente_de) + filas_sinteticas())
    w.to_pickle(SALIDA / "palabras_etiquetadas.pkl")
    print(f"palabras: {len(w):,} | instancias: {w.groupby(['fuente', 'instancia']).ngroups:,}")

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    por_fuente = tabla(w, ["fuente"])
    total = tabla(w.assign(g="TOTAL (sin sintético)").query("fuente != 'sintetico'"), ["g"])
    print("\n== POR FUENTE ==\n", pd.concat([por_fuente, total]).to_string())
    por_fuente.to_csv(SALIDA / "por_fuente.csv")

    pel = tabla(w[w["fuente"] == "subtitulos"], ["grupo"])
    pel["¿ / ?"] = (pel["n ¿"] / pel["n ?"]).round(2)
    pel = pel.sort_values("¿ %")
    print("\n== POR PELÍCULA (subtítulos), ordenadas por ¿ % ==\n", pel[["palabras", "instancias", "¿ %", "? %", "n ¿", "n ?", "¿ / ?", "coma %", "punto %", "cap1 %", "% instancias con pregunta"]].to_string())
    pel.to_csv(SALIDA / "por_pelicula.csv")

    q = preguntas(w)
    q.to_csv(SALIDA / "preguntas.csv", index=False)
    cerr = q[q["cerrada"] & q["largo"].notna()].copy()
    cerr["largo_bin"] = pd.cut(cerr["largo"], [0, 1, 2, 4, 7, 12, 1000], labels=["1", "2", "3-4", "5-7", "8-12", "13+"])
    print("\n== LARGO DE LAS PREGUNTAS (palabras de ¿ a ?), % dentro de cada fuente ==")
    print((100 * pd.crosstab(cerr["fuente"], cerr["largo_bin"], normalize="index")).round(1).to_string())
    print("\n== DÓNDE ABRE EL ¿ (% dentro de cada fuente) ==")
    print((100 * pd.crosstab(q["fuente"], q["abre"], normalize="index")).round(1).to_string())
    print("\ncantidad de preguntas por fuente:", q["fuente"].value_counts().to_dict())


if __name__ == "__main__":
    main()
