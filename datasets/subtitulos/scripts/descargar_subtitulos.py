import argparse
import csv
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

BASE = "https://api.opensubtitles.com/api/v1"
USER_AGENT = "TP2-AA-puntuacion v0.1"
RAIZ = Path(__file__).resolve().parent.parent  # datasets/subtitulos/
MANIFEST_COLUMNS = ["titulo_buscado", "titulo", "anio", "imdb_id", "idioma", "file_id",
                    "release", "hearing_impaired", "download_count", "archivo"]


def cargar_env():
    # el .env se busca en la carpeta del script y hacia arriba (en el repo vive en la raíz)
    ruta = next((c / ".env" for c in Path(__file__).resolve().parents if (c / ".env").exists()), None)
    if ruta is None:
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, _, valor = linea.partition("=")
            # setdefault: una variable ya cargada en la terminal gana sobre el .env
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def nueva_sesion():
    s = requests.Session()
    s.headers.update({
        "Api-Key": os.environ["OPENSUBTITLES_API_KEY"],
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    for intento in range(3):
        r = s.post(f"{BASE}/login", json={
            "username": os.environ["OPENSUBTITLES_USER"],
            "password": os.environ["OPENSUBTITLES_PASS"],
        })
        if r.status_code != 429 or intento == 2:
            break
        time.sleep(int(r.headers.get("Retry-After", 5)))  # no insistir con logins en paralelo contra la cuenta
    r.raise_for_status()
    s.headers["Authorization"] = f"Bearer {r.json()['token']}"
    return s


def llamar(s, metodo, url, **kw):
    for _ in range(5):
        r = s.request(metodo, url, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(int(r.headers.get("Retry-After", 5)))
    r.raise_for_status()


def palabras(texto):
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return set(re.sub(r"[^a-z0-9]+", " ", texto.lower()).split())


def coincidencia(titulo, a):
    feat = a.get("feature_details") or {}
    texto = " ".join(str(x) for x in (a.get("slug"), a.get("release"), feat.get("title"), feat.get("movie_name")))
    buscadas = palabras(titulo)
    return len(buscadas & palabras(texto)) / len(buscadas) if buscadas else 0


DESCARTAR = re.compile(r"dub|forced|forzad|foreign|narrat|doblad", re.I)
# fansubs argentinos que transcriben el audio (conservan el voseo) y subs oficiales de disco o autogenerados (suelen venir neutralizados)
FANSUB_AR = re.compile(r"clan-?sud|argenteam|ripead|estrenosdivx", re.I)
NEUTRALIZADO = re.compile(r"yts|yify|artsubs|horizon|nodlabs|desmultiplex|allsubs|lokoto|bdrip|blu-?ray", re.I)


def puntaje(a):
    """Puntaje de un subtítulo para elegir uno por película; None si hay que descartarlo.

    Descarta: traducidos por máquina, de varios CD (el script bajaría solo files[0], media película),
    otro idioma y releases doblados / de partes en otro idioma. Suma por uploader confiable, release
    de fansub argentino (clan-sud, aRGENTeaM...: transcriben el audio y conservan el voseo) y descargas; resta para
    sordos (traen [música], etc.) y para releases de disco/autogenerados (BluRay, ArtSubs, YTS...), que en la
    tanda del 2026-11-16 salieron 6 de 6 en español neutro. Es una heurística con pocos datos: siempre verificar
    con analizar_subtitulos.py (voseo_por_mil, apertura_sobre_cierre).
    `foreign_parts_only` de la API no sirve: acepta el filtro pero ningún subtítulo viene marcado.
    """
    if a.get("ai_translated") or a.get("machine_translated"):
        return None
    if len(a.get("files") or []) != 1 or (a.get("nb_cd") or 1) > 1:
        return None
    if a.get("language") not in ("es", "ea"):
        return None
    texto = f"{a.get('release') or ''} {a.get('comments') or ''} {a['files'][0].get('file_name') or ''}"
    if DESCARTAR.search(texto):
        return None
    p = math.log10(1 + (a.get("download_count") or 0))
    p += 1.0 if a.get("from_trusted") else 0
    p += 1.0 if FANSUB_AR.search(texto) else 0
    p -= 1.0 if NEUTRALIZADO.search(texto) else 0
    p -= 1.5 if a.get("hearing_impaired") else 0
    p += 0.2 if (a.get("fps") or 0) in (23.976, 24.0, 25.0) else 0
    return round(p, 2)


def buscar(s, titulo, anio, imdb_fijo, idiomas, debug):
    params = {"languages": idiomas, "type": "movie"}
    params.update({"imdb_id": imdb_fijo} if imdb_fijo else {"query": titulo})
    r = llamar(s, "GET", f"{BASE}/subtitles", params=params)
    resultados = r.json().get("data", [])
    if debug and resultados:
        print(json.dumps(resultados[0], indent=1, ensure_ascii=False))

    candidatos = []
    for item in resultados:
        a = item.get("attributes", {})
        feat = a.get("feature_details") or {}
        if not a.get("files") or a.get("ai_translated") or a.get("machine_translated"):
            continue
        if not imdb_fijo and anio and feat.get("year") and abs(int(feat["year"]) - anio) > 1:
            continue
        candidatos.append(a)
    if not candidatos:
        return [], False

    # una misma búsqueda puede mezclar películas distintas (remakes, títulos repetidos):
    # gana la que mejor coincide en título y, a igual coincidencia, la más descargada
    por_imdb = {}
    for a in candidatos:
        imdb = (a.get("feature_details") or {}).get("imdb_id")
        c, d = por_imdb.get(imdb, (0, 0))
        por_imdb[imdb] = (max(c, coincidencia(titulo, a)), max(d, a.get("download_count", 0)))
    imdb, (mejor_coincidencia, _) = max(por_imdb.items(), key=lambda kv: kv[1])
    candidatos = [a for a in candidatos if (a.get("feature_details") or {}).get("imdb_id") == imdb]

    # un solo archivo por película (cada descarga gasta cupo): el de mejor puntaje entre todos los idiomas pedidos
    puntuados = [(puntaje(a), a) for a in candidatos]
    puntuados = [(p, a) for p, a in puntuados if p is not None]
    if not puntuados:
        return [], False
    _, mejor = max(puntuados, key=lambda x: x[0])
    return [mejor], (not imdb_fijo and mejor_coincidencia < 1)


def descargar(s, file_id):
    r = llamar(s, "POST", f"{BASE}/download", json={"file_id": file_id, "sub_format": "srt"}).json()
    contenido = requests.get(r["link"]).content
    return contenido, r.get("remaining")


def nombre_seguro(texto):
    return re.sub(r'[<>:"/\\|?*]', "", texto).strip()


def leer_peliculas(ruta):
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        partes = [x.strip() for x in linea.split(";")] + ["", ""]
        yield partes[0], int(partes[1]) if partes[1] else None, partes[2] or None


def main():
    cargar_env()
    p = argparse.ArgumentParser()
    p.add_argument("--peliculas", default=str(RAIZ / "listas" / "peliculas.txt"))
    p.add_argument("--salida", default=str(RAIZ / "subs"))
    p.add_argument("--idiomas", default="es,ea,sp")
    p.add_argument("--max-descargas", type=int, default=20, help="tope de descargas de esta corrida (el cupo diario es 20)")
    p.add_argument("--dry-run", action="store_true", help="solo busca y lista, no descarga")
    p.add_argument("--debug", action="store_true", help="imprime el primer resultado crudo de cada búsqueda")
    args = p.parse_args()

    salida = Path(args.salida)
    salida.mkdir(exist_ok=True)
    manifest = salida / "manifest.csv"
    if not manifest.exists() and not args.dry_run:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MANIFEST_COLUMNS)

    s = nueva_sesion()
    cupo = llamar(s, "GET", f"{BASE}/infos/user").json()["data"].get("remaining_downloads")
    print(f"cupo restante: {cupo}")
    bajadas = 0
    tope = min(args.max_descargas, cupo if cupo is not None else args.max_descargas)
    for titulo, anio, imdb_fijo in leer_peliculas(args.peliculas):
        if not args.dry_run and bajadas >= tope:
            print(f"tope de {tope} descargas alcanzado")
            break
        elegidos, dudosa = buscar(s, titulo, anio, imdb_fijo, args.idiomas, args.debug)
        time.sleep(0.5)
        if not elegidos:
            print(f"[sin resultados] {titulo} ({anio})")
            continue

        feat = elegidos[0].get("feature_details") or {}
        resumen = ", ".join(f"{a['language']}:{a.get('download_count', 0)}" for a in elegidos)
        marca = "[revisar] " if dudosa else ""
        print(f"{marca}{titulo} -> {feat.get('title')} ({feat.get('year')}) imdb={feat.get('imdb_id')} -> {resumen}")
        if args.dry_run:
            continue
        if dudosa:
            print("  saltada: título dudoso, fijá el imdb_id en peliculas.txt")
            continue

        for a in elegidos:
            file_id = a["files"][0]["file_id"]
            sufijo = f"[{a['language']}-{file_id}].srt"
            if any(p.name.endswith(sufijo) for p in salida.iterdir()):
                continue
            destino = salida / f"{nombre_seguro(titulo)} ({feat.get('year')}) {sufijo}"
            try:
                contenido, restantes = descargar(s, file_id)
            except requests.HTTPError as e:
                print(f"descarga fallida ({e.response.status_code}): {e.response.text[:200]}")
                return
            bajadas += 1
            destino.write_bytes(contenido)
            with manifest.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    titulo, feat.get("title"), feat.get("year"), feat.get("imdb_id"),
                    a["language"], file_id, a.get("release"), a.get("hearing_impaired"),
                    a.get("download_count"), destino.name,
                ])
            print(f"  guardado {destino.name} (quedan {restantes} descargas)")
            if restantes is not None and restantes <= 0:
                print("cupo diario agotado")
                return
            time.sleep(0.5)


if __name__ == "__main__":
    main()
