import argparse
import csv
import re
from pathlib import Path

EN = {"the", "you", "and", "to", "is", "it", "that", "of", "in", "he", "what", "i"}
VOSEO = {"vos", "tenés", "querés", "podés", "sabés", "decís", "hacés", "venís", "andás", "sos", "che",
         "boludo", "pelotudo", "pibe", "laburo", "quilombo", "mirá", "decime", "fijate", "dale", "pendejo",
         "guita", "mina", "vení", "hacé", "dejá", "escuchá", "pará", "andá", "sentate", "bondi", "chabón",
         "posta"}
COLUMNAS = ["archivo", "idioma", "codificacion", "bloques", "palabras", "apertura", "cierre", "apertura_sobre_cierre",
            "puntos", "comas", "lineas_con_mayuscula_pct", "guiones", "corchetes", "voseo", "voseo_por_mil",
            "top_voseo"]


def leer(ruta):
    datos = Path(ruta).read_bytes()
    try:
        return datos.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return datos.decode("cp1252", errors="replace"), "cp1252"


def detectar_idioma(texto, palabras):
    # ñ ¿ ¡ solo existen en español y ã õ ç solo en portugués; las palabras funcionales comunes
    # (que, de, no...) no sirven porque las comparten
    es = sum(texto.count(c) for c in "ñÑ¿¡")
    pt = sum(texto.count(c) for c in "ãõçÃÕÇ")
    en = sum(p in EN for p in palabras) / max(len(palabras), 1)
    if pt > 20 and pt > es:
        return "pt"
    if es > 20:
        return "es"
    if en > 0.15:
        return "en"
    return "?"


def analizar(ruta):
    texto, codificacion = leer(ruta)
    lineas = [l.strip() for l in texto.splitlines()]
    dialogo = [l for l in lineas if l and not l.isdigit() and "-->" not in l]
    palabras = re.findall(r"[a-záéíóúñü]+", " ".join(dialogo).lower())
    marcas = {}
    for p in palabras:
        if p in VOSEO:
            marcas[p] = marcas.get(p, 0) + 1
    apertura, cierre = texto.count("¿"), texto.count("?")
    return {
        "archivo": Path(ruta).name,
        "idioma": detectar_idioma(texto, palabras),
        "codificacion": codificacion,
        "bloques": sum("-->" in l for l in lineas),
        "palabras": len(palabras),
        "apertura": apertura,
        "cierre": cierre,
        "apertura_sobre_cierre": round(apertura / cierre, 2) if cierre else "",
        "puntos": texto.count("."),
        "comas": texto.count(","),
        "lineas_con_mayuscula_pct": round(100 * sum(l[:1].isupper() for l in dialogo) / max(len(dialogo), 1)),
        "guiones": sum(l.startswith("-") for l in dialogo),
        "corchetes": sum("[" in l for l in dialogo),
        "voseo": sum(marcas.values()),
        "voseo_por_mil": round(1000 * sum(marcas.values()) / max(len(palabras), 1), 1),
        "top_voseo": " ".join(f"{p}:{n}" for p, n in sorted(marcas.items(), key=lambda kv: -kv[1])[:5]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rutas", nargs="+", help="archivos .srt o carpetas")
    p.add_argument("--csv", help="guarda la tabla completa en este archivo")
    args = p.parse_args()

    archivos = []
    for ruta in args.rutas:
        ruta = Path(ruta)
        archivos += sorted(ruta.rglob("*.srt")) if ruta.is_dir() else [ruta]
    filas = [analizar(a) for a in archivos]

    print(f"{'archivo':<52} {'idi':<4} {'bloq':>5} {'pal':>6} {'¿':>4} {'?':>4} {'¿/?':>5} {'vos':>4} {'‰':>5} {'may%':>5}  top voseo")
    for f in filas:
        print(f"{f['archivo'][:52]:<52} {f['idioma']:<4} {f['bloques']:>5} {f['palabras']:>6} {f['apertura']:>4} "
              f"{f['cierre']:>4} {f['apertura_sobre_cierre']:>5} {f['voseo']:>4} {f['voseo_por_mil']:>5} "
              f"{f['lineas_con_mayuscula_pct']:>5}  {f['top_voseo']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)


if __name__ == "__main__":
    main()
