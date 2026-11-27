import argparse
import csv
import sys
import time
from pathlib import Path

import descargar_subtitulos as d

ES_PREFIJOS = ("es", "ea", "sp")
COL_INTENTOS = ["pelicula", "anio_esperado", "idioma_titulo", "titulo_consultado", "anio_consultado", "resultados",
                "mejor_title_os", "mejor_original_title", "mejor_anio", "similitud", "aceptado", "imdb_id"]
COL_RESUMEN = ["pelicula", "anio_esperado", "ya_en_listas_previas", "titulos_es_probados", "titulos_en_probados",
               "title_os", "original_title_os", "anio_os", "imdb_id", "subs_ea", "subs_es", "otros_es", "idiomas_totales",
               "titulo_que_matcheo", "idioma_que_matcheo", "intentos", "estado"]


def leer_lista(ruta):
    vistos = set()
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.startswith("#"):
            continue
        es, anio, en = [x.strip() for x in linea.split("|")]
        titulos_es = [t.strip() for t in es.split(";") if t.strip()]
        titulos_en = [t.strip() for t in en.split(";") if t.strip()]
        if (titulos_es[0], anio) in vistos:
            continue
        vistos.add((titulos_es[0], anio))
        yield titulos_es, int(anio), titulos_en


def titulos_de_listas_previas(rutas):
    previos = set()
    for ruta in rutas:
        if Path(ruta).exists():
            for titulo, _, _ in d.leer_peliculas(ruta):
                previos.add(frozenset(d.palabras(titulo)))
    return previos


def similitud(titulos, a):
    mejor = 0
    for t in titulos:
        buscadas = d.palabras(t)
        for campo in (a.get("title"), a.get("original_title")):
            pal = d.palabras(campo)
            if buscadas and pal:
                mejor = max(mejor, len(buscadas & pal) / max(len(buscadas), len(pal)))
    return mejor


def puntaje(a, titulos_es, titulos_en):
    return similitud(titulos_es, a), similitud(titulos_en, a)


def consultar(s, titulo, anio):
    params = {"query": titulo, "type": "movie"}
    if anio:
        params["year"] = anio
    r = d.llamar(s, "GET", f"{d.BASE}/features", params=params).json()
    time.sleep(0.35)
    return [it.get("attributes", {}) for it in r.get("data", [])]


def resolver(s, titulos_es, anio, titulos_en, registrar):
    """Devuelve (feature, estado, titulo_que_matcheo, idioma_que_matcheo).

    Acepta de inmediato solo si el título en español coincide con title/original_title de OpenSubtitles.
    Si solo coincide un título en inglés (puede ser otra película con el mismo nombre) lo guarda como
    candidato "dudoso" y sigue buscando; si no aparece nada mejor, devuelve ese candidato.
    """
    todos = [("en", t) for t in titulos_en] + [("es", t) for t in titulos_es]
    dudoso = None
    # año exacto, luego ±1 y por último sin año (aceptando hasta ±2) por si el año de memoria está mal
    for anio_consulta in (anio, anio - 1, anio + 1, None):
        umbral = 0.9 if anio_consulta is None else 0.8
        for idioma, titulo in todos:
            cands = consultar(s, titulo, anio_consulta)
            if anio_consulta is None:
                cands_val = [a for a in cands if a.get("year") and abs(int(a["year"]) - anio) <= 2]
            else:
                cands_val = cands
            puntuados = [(puntaje(a, titulos_es, titulos_en), a) for a in cands_val]
            mejor_es = max(puntuados, key=lambda x: x[0][0], default=None)
            mejor_en = max(puntuados, key=lambda x: x[0][1], default=None)
            if mejor_es and mejor_es[0][0] >= umbral:
                veredicto, a, sim = "si", mejor_es[1], mejor_es[0][0]
            elif mejor_en and mejor_en[0][1] >= umbral:
                veredicto, a, sim = "solo_titulo_en", mejor_en[1], mejor_en[0][1]
            else:
                veredicto, a, sim = "no", (mejor_es or (None, None))[1], (mejor_es[0][0] if mejor_es else 0)
            registrar([titulos_es[0], anio, idioma, titulo, anio_consulta or "sin año", len(cands),
                       (a or {}).get("title", ""), (a or {}).get("original_title", ""), (a or {}).get("year", ""),
                       f"{sim:.2f}", veredicto, (a or {}).get("imdb_id", "") if veredicto != "no" else ""])
            if veredicto == "si":
                return a, "ok", titulo, idioma
            if veredicto == "solo_titulo_en" and not dudoso:
                dudoso = (a, "dudoso", titulo, idioma)
    return dudoso or (None, "sin_match", "", "")


def main():
    d.cargar_env()
    p = argparse.ArgumentParser()
    p.add_argument("--lista", default=str(d.RAIZ / "listas" / "lista_peliculas_ar.txt"))
    p.add_argument("--intentos", default=str(d.RAIZ / "resultados" / "titulos_probados.csv"))
    p.add_argument("--resumen", default=str(d.RAIZ / "resultados" / "peliculas_resueltas.csv"))
    p.add_argument("--previas", nargs="*",
                   default=[str(d.RAIZ / "listas" / "peliculas.txt"), str(d.RAIZ / "listas" / "peliculas_candidatas.txt")],
                   help="listas anteriores, solo para marcar cuáles ya se habían probado")
    p.add_argument("--agregar", action="store_true",
                   help="agrega a --intentos y --resumen en vez de pisarlos y saltea las películas que ya están en --resumen")
    args = p.parse_args()

    previos = titulos_de_listas_previas(args.previas)
    hechas = set()
    if args.agregar and Path(args.resumen).exists():
        with open(args.resumen, encoding="utf-8") as f:
            hechas = {(r["pelicula"], r["anio_esperado"]) for r in csv.DictReader(f)}
    # los CSV nuevos o vacíos llevan encabezado; si se agrega a uno existente no se repite
    con_encabezado = [not (args.agregar and Path(r).exists() and Path(r).stat().st_size > 0) for r in (args.intentos, args.resumen)]
    s = d.nueva_sesion()
    with open(args.intentos, "a" if args.agregar else "w", newline="", encoding="utf-8") as fi, \
            open(args.resumen, "a" if args.agregar else "w", newline="", encoding="utf-8") as fr:
        wi, wr = csv.writer(fi), csv.writer(fr)
        if con_encabezado[0]:
            wi.writerow(COL_INTENTOS)
        if con_encabezado[1]:
            wr.writerow(COL_RESUMEN)
        for titulos_es, anio, titulos_en in leer_lista(args.lista):
            if (titulos_es[0], str(anio)) in hechas:
                continue
            intentos = []

            def registrar(fila):
                intentos.append(fila)
                wi.writerow(fila)
                fi.flush()

            a, estado, titulo_match, idioma_match = resolver(s, titulos_es, anio, titulos_en, registrar)
            previa = "si" if any(frozenset(d.palabras(t)) in previos for t in titulos_es) else "no"
            base = [titulos_es[0], anio, previa, " | ".join(titulos_es), " | ".join(titulos_en)]
            if not a:
                wr.writerow(base + [""] * 10 + [len(intentos), "sin_match"])
                print(f"[sin match] {titulos_es[0]} ({anio}) tras {len(intentos)} intentos", flush=True)
            else:
                sc = a.get("subtitles_counts") or {}
                otros = " ".join(f"{k}:{v}" for k, v in sc.items() if k.startswith(ES_PREFIJOS) and k not in ("es", "ea"))
                wr.writerow(base + [a.get("title"), a.get("original_title"), a.get("year"), a.get("imdb_id"),
                                    sc.get("ea", 0), sc.get("es", 0), otros, len(sc), titulo_match, idioma_match,
                                    len(intentos), estado])
                print(f"{'[dudoso] ' if estado == 'dudoso' else ''}{titulos_es[0]} ({anio}) -> {a.get('title')} | "
                      f"{a.get('original_title')} ({a.get('year')}) "
                      f"imdb={a.get('imdb_id')} ea={sc.get('ea', 0)} es={sc.get('es', 0)} "
                      f"[matcheó {idioma_match}: '{titulo_match}', {len(intentos)} intentos]", flush=True)
            fr.flush()
    print("listo")


if __name__ == "__main__":
    sys.exit(main())
