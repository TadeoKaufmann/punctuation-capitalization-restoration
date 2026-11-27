import argparse
import csv
import time

import requests

import descargar_subtitulos as d

COLUMNAS = ["pelicula_buscada", "dudosa", "titulo", "anio", "imdb_id", "idioma", "file_id", "from_trusted",
            "descargas", "subido", "uploader", "release"]


def imdb_de(a):
    return (a.get("feature_details") or {}).get("imdb_id")


def candidatos(s, titulo, anio, imdb_fijo, idiomas):
    params = {"languages": idiomas, "type": "movie"}
    params.update({"imdb_id": imdb_fijo} if imdb_fijo else {"query": titulo})
    resultados = d.llamar(s, "GET", f"{d.BASE}/subtitles", params=params).json().get("data", [])
    out = []
    for item in resultados:
        a = item.get("attributes", {})
        feat = a.get("feature_details") or {}
        if not a.get("files") or a.get("ai_translated") or a.get("machine_translated"):
            continue
        if not imdb_fijo and anio and feat.get("year") and abs(int(feat["year"]) - anio) > 1:
            continue
        out.append(a)
    return out


def main():
    d.cargar_env()
    p = argparse.ArgumentParser()
    p.add_argument("--peliculas", default=str(d.RAIZ / "listas" / "peliculas_candidatas.txt"))
    p.add_argument("--idiomas", default="es,ea")
    p.add_argument("--salida", default=str(d.RAIZ / "resultados" / "candidatas.csv"))
    args = p.parse_args()

    s = d.nueva_sesion()
    filas = []
    buscadas = con_resultados = 0
    for titulo, anio, imdb_fijo in d.leer_peliculas(args.peliculas):
        buscadas += 1
        try:
            cands = candidatos(s, titulo, anio, imdb_fijo, args.idiomas)
        except requests.HTTPError as e:
            print(f"[error {e.response.status_code}] {titulo}")
            continue
        time.sleep(0.5)
        if not cands:
            continue

        # una búsqueda puede mezclar películas distintas: gana la que mejor coincide en título
        por_imdb = {}
        for a in cands:
            c, n = por_imdb.get(imdb_de(a), (0, 0))
            por_imdb[imdb_de(a)] = (max(c, d.coincidencia(titulo, a)), max(n, a.get("download_count", 0)))
        imdb, (puntaje, _) = max(por_imdb.items(), key=lambda kv: kv[1])
        dudosa = not imdb_fijo and puntaje < 1
        elegidas = sorted((a for a in cands if imdb_de(a) == imdb), key=lambda a: -a.get("download_count", 0))

        con_resultados += 1
        for a in elegidas:
            feat = a.get("feature_details") or {}
            fila = [titulo, "revisar" if dudosa else "", feat.get("title"), feat.get("year"), imdb, a.get("language"),
                    a["files"][0]["file_id"], a.get("from_trusted"), a.get("download_count"),
                    str(a.get("upload_date"))[:10], (a.get("uploader") or {}).get("name"), a.get("release")]
            filas.append(fila)
            print(f"{'[revisar] ' if dudosa else ''}{titulo} -> {feat.get('title')} ({feat.get('year')}) imdb={imdb} | "
                  f"{a.get('language')} file_id={fila[6]} trusted={fila[7]} descargas={fila[8]} "
                  f"uploader={fila[10]} | {str(fila[11])[:40]}")

    Path(args.salida).parent.mkdir(exist_ok=True)
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        w.writerows(filas)
    print(f"\n{buscadas} películas buscadas, {con_resultados} con subtítulos en {args.idiomas} -> {args.salida}")


if __name__ == "__main__":
    main()
