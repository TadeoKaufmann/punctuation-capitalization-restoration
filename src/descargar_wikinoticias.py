"""Descarga notas de es.wikinews.org (texto plano) de categorias sobre paises y organizaciones.

Motivacion: el corpus tiene muy pocos ejemplos de capitalizacion clase 2 (mixta, tipo UPyD/iPhone)
y clase 3 (siglas, tipo ONU/OTAN). Wikinoticias aporta nombres de paises, organizaciones y siglas
en contexto.

Salida: datasets/wikinoticias/raw/notas.jsonl  (una nota por linea: pageid, titulo, categoria, texto)
Es reanudable: si se corta, al volver a correr saltea las notas ya bajadas.

Uso:  python src/descargar_wikinoticias.py [--max-por-categoria 400]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://es.wikinews.org/w/api.php"
UA = "tp-aa2-corpus-academico/0.1 (trabajo practico universitario)"
CATEGORIAS = [
    "ONU", "Diplomacia", "Unión Europea", "Mundo",
    "Economía y Negocios", "Democracia y elecciones", "Política",
]
SALIDA = Path(__file__).resolve().parent.parent / "datasets" / "wikinoticias" / "raw" / "notas.jsonl"


def api(**params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    for intento in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:  # incluye 429 (rate limit) y respuestas no-JSON
            time.sleep(15 * (intento + 1))
    raise RuntimeError("fallo la API: " + url)


def miembros(categoria, maximo):
    """Devuelve [(pageid, titulo)] de las notas de la categoria (namespace 0)."""
    out, cont = [], {}
    while len(out) < maximo:
        d = api(action="query", list="categorymembers", cmtitle=f"Categoría:{categoria}",
                cmnamespace=0, cmlimit=500, **cont)
        out += [(m["pageid"], m["title"]) for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = d["continue"]
    return out[:maximo]


def limpiar_wikitexto(w):
    """Quita plantillas, enlaces, tablas, referencias y marcado; deja texto plano."""
    # {{w|Entidad}} y {{w|Articulo|texto visible}} son los enlaces a Wikipedia: llevan los nombres propios
    w = re.sub(r"\{\{(?:w|wikipedia)\|([^{}|]*)(?:\|([^{}]*))?\}\}",
               lambda m: m.group(2) or m.group(1), w, flags=re.I)
    for _ in range(5):  # el resto de las plantillas (fecha, fuente, etc.) se descarta
        w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    w = re.sub(r"\{\|.*?\|\}", "", w, flags=re.S)                          # tablas
    w = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", w, flags=re.S)      # referencias
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)                           # comentarios
    w = re.sub(r"\[\[(?:Categoría|Category|Archivo|File|Imagen|Image):[^\]]*\]\]", "", w, flags=re.I)
    w = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", w)                # [[a|b]] -> b
    w = re.sub(r"\[https?://\S+ ([^\]]*)\]|\[https?://\S+\]", r"\1", w)    # enlaces externos
    w = re.sub(r"<[^>]+>", "", w)                                          # html
    w = re.sub(r"'{2,}", "", w)                                            # negrita/cursiva
    w = re.sub(r"^[=*#:;].*$", "", w, flags=re.M)                          # titulos y listas
    w = re.sub(r"[ \t]+", " ", w)
    w = re.sub(r"\n{2,}", "\n", w)
    return w.strip()


def textos(pageids):
    """Wikitexto de hasta 50 paginas en un solo pedido -> {pageid: texto_limpio}."""
    d = api(action="query", prop="revisions", rvprop="content", rvslots="main",
            pageids="|".join(map(str, pageids)))
    out = {}
    for pid, pg in d["query"]["pages"].items():
        revs = pg.get("revisions")
        if revs:
            out[int(pid)] = limpiar_wikitexto(revs[0]["slots"]["main"]["*"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-por-categoria", type=int, default=400)
    a = ap.parse_args()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if SALIDA.exists():
        hechos = {json.loads(l)["pageid"] for l in SALIDA.open(encoding="utf-8")}
    with SALIDA.open("a", encoding="utf-8") as f:
        for cat in CATEGORIAS:
            lista = [m for m in miembros(cat, a.max_por_categoria) if m[0] not in hechos]
            print(f"{cat}: {len(lista)} notas nuevas", flush=True)
            for i in range(0, len(lista), 50):
                lote = lista[i:i + 50]
                tx = textos([pid for pid, _ in lote])
                for pid, titulo in lote:
                    hechos.add(pid)
                    if tx.get(pid, "").strip():
                        f.write(json.dumps({"pageid": pid, "titulo": titulo, "categoria": cat,
                                            "texto": tx[pid]}, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(2)
    print("listo:", len(hechos), "notas en", SALIDA)


if __name__ == "__main__":
    main()
