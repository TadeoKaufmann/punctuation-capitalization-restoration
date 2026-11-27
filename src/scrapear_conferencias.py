"""
Arma fragmentos (1 a 3 oraciones) a partir de las conferencias de prensa de
https://www.casarosada.gob.ar/informacion/conferencias

Pipeline:
  1. listar_conferencias(): recorre el listado paginado (?start=0,40,80,...).
  2. descargar(): baja cada conferencia (con caché en disco y pausa entre pedidos).
  3. extraer_turnos(): se queda SOLO con los turnos "ORADOR.- texto" (preguntas y
     respuestas); ignora título, fecha, iframes y el discurso inicial.
  4. limpiar_turno() / oraciones_validas(): saca aclaraciones entre paréntesis, corta en
     oraciones y descarta las que tienen puntuación fuera de , . ? o números largos.
  5. armar_fragmentos(): parte cada turno en fragmentos de 1 a 3 oraciones consecutivas.

Salida: fragmentos_conferencias.csv (columnas Indice, Fragmento, Conferencia), mismo
formato que fragmentos.csv. Después se pasa por armar_instancias.armar_dataset.
"""
import argparse
import html
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from rutas import DATASETS, SALIDAS

BASE = "https://www.casarosada.gob.ar"
LISTADO = BASE + "/informacion/conferencias"
CACHE = SALIDAS / "cache_conferencias"
HEADERS = {"User-Agent": "Mozilla/5.0 (trabajo practico universitario de aprendizaje automatico)"}
PAUSA = 1.0                 # segundos entre pedidos
POR_PAGINA = 40
INDICE_INICIAL = 100001     # lejos de los Indice de Paenza (1..~3400) para poder juntar ambos datasets

MAX_DIGITOS_PALABRA = 4     # se descarta la oración si alguna palabra tiene MÁS de 4 dígitos
MAX_ORACIONES = 3

# Etiqueta de orador en cualquier parte del párrafo (varios turnos pueden venir en un mismo <p>).
# Variantes reales: "ADORNI.- ", "ADORNI. - ", "ADORNI.– ", "PRESIDENTE- ", "PERIODISTA - ", "PERIODISTA (Nombre, Medio).- ".
# Sin punto solo vale al inicio del párrafo o tras fin de oración (evita "la ONU - para").
RE_ETIQUETA = re.compile(r"(?<![A-Za-zÁÉÍÓÚÑáéíóúñ0-9])([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 ./'\-]{2,40}?)(?:\s*\([^)]*\))?\s*(\.?)\s*[-–—]\s*")
RE_PARENTESIS = re.compile(r"\([^)]*\)|\[[^\]]*\]")
RE_PARENTESIS_SIN_CERRAR = re.compile(r"\([^)]*$")
# Título repetido dentro del cuerpo: "... desde Casa Rosada 10-12-2025"
RE_TITULO = re.compile(r"(?:conferencia de prensa|anuncio|desde casa rosada).*\d{1,2}-\d{1,2}-\d{2,4}\s*$", re.I)
# Puntuación que las etiquetas (¿ , . ?) no cubren: la oración se descarta
RE_PUNT_AJENA = re.compile(r"[…:;!¡]|\.\.")
COMILLAS = str.maketrans("", "", "\"“”«»‘’'")

# Abreviaturas cuyo punto NO termina la oración
ABREVIATURAS = ["Sr", "Sra", "Srta", "Dr", "Dra", "Lic", "Ing", "Gral", "Cnel", "Av", "Prof", "Arq",
                "Cdor", "Dip", "Sen", "Mons", "Pte", "Ud", "Uds", "etc", "Nº", "N°", "EE", "UU", "S.A"]
RE_ABREV = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABREVIATURAS) + r")\.")
RE_INICIAL = re.compile(r"\b([A-ZÁÉÍÓÚÑ])\.(?=\s*[A-ZÁÉÍÓÚÑ])")   # "J. M. Milei"
RE_CORTE = re.compile(r"(?<=[.?])\s+(?=[¿A-ZÁÉÍÓÚÑ0-9])")
MARCA = "\u0001"            # reemplaza temporalmente al punto de una abreviatura


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def descargar(url: str, cache: Path = CACHE) -> str:
    """Baja `url` (o la lee del caché). Pausa PAUSA segundos si hizo un pedido real."""
    cache.mkdir(exist_ok=True)
    archivo = cache / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")).strip("_")[:150] + ".html")
    if archivo.exists():
        return archivo.read_text(encoding="utf-8")
    for intento in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if intento == 2:
                raise
            print(f"  reintento {url}: {e}")
            time.sleep(3)
    r.encoding = "utf-8"
    archivo.write_text(r.text, encoding="utf-8")
    time.sleep(PAUSA)
    return r.text


def listar_conferencias() -> list[str]:
    """URLs de todas las conferencias del listado, sin repetir y en orden de aparición."""
    urls, vistas = [], set()
    for start in range(0, 2000, POR_PAGINA):
        pagina = descargar(f"{LISTADO}?start={start}" if start else LISTADO)
        nuevas = [
            BASE + h for h in re.findall(r'href="(/informacion/conferencias/\d+-[^"?#]+)"', pagina)
            if h not in vistas
        ]
        if not nuevas:
            break
        for h in nuevas:
            vistas.add(h[len(BASE):])
        urls.extend(dict.fromkeys(nuevas))
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# Extracción de turnos
# ---------------------------------------------------------------------------
def es_etiqueta(texto: str, m: re.Match) -> bool:
    """Con punto ("ADORNI.-") siempre; sin punto solo al inicio o después de . ? ! … )."""
    if m.group(2) == ".":
        return True
    previo = texto[:m.start()].rstrip()
    return not previo or previo[-1] in ".?!…)"


def extraer_turnos(pagina: str) -> list[tuple[str, str]]:
    """Lista de (orador, texto) de los párrafos 'ORADOR.- ...'.

    Un párrafo sin orador que viene después de un turno es continuación del mismo turno.
    Todo lo anterior al primer turno (título, discurso inicial) se ignora.
    """
    soup = BeautifulSoup(pagina, "html.parser")
    cuerpo = soup.find("article") or soup
    turnos: list[list[str]] = []
    for p in cuerpo.find_all("p"):
        if p.find("iframe"):
            continue
        texto = " ".join(html.unescape(p.get_text(" ")).replace("\xa0", " ").split())
        if not texto or RE_TITULO.search(texto):
            continue
        marcas = [m for m in RE_ETIQUETA.finditer(texto) if es_etiqueta(texto, m)]
        if marcas and marcas[0].start() > 0 and turnos:
            turnos[-1][1] += " " + texto[:marcas[0].start()]      # cola del turno anterior
        for i, m in enumerate(marcas):
            fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
            turnos.append([m.group(1).strip(), texto[m.end():fin]])
        if not marcas and turnos:
            turnos[-1][1] += " " + texto
    return [(o, t) for o, t in turnos]


def tiene_periodistas(turnos: list[tuple[str, str]]) -> bool:
    return any(o.upper().startswith("PERIODISTA") and limpiar_turno(t) for o, t in turnos)


# ---------------------------------------------------------------------------
# Limpieza y oraciones
# ---------------------------------------------------------------------------
def limpiar_turno(texto: str) -> str:
    """Saca aclaraciones entre paréntesis/corchetes y comillas; normaliza espacios."""
    texto = RE_PARENTESIS.sub(" ", texto)
    texto = RE_PARENTESIS_SIN_CERRAR.sub(" ", texto)
    texto = texto.translate(COMILLAS)
    texto = re.sub(r"\s+([,.?])", r"\1", texto)      # "hola ." -> "hola."
    return " ".join(texto.split())


def dividir_oraciones(texto: str) -> list[str]:
    texto = RE_ABREV.sub(lambda m: m.group(1) + MARCA, texto)
    texto = RE_INICIAL.sub(lambda m: m.group(1) + MARCA, texto)
    return [o.replace(MARCA, ".").strip() for o in RE_CORTE.split(texto) if o.strip()]


def oracion_valida(o: str) -> bool:
    if not o.endswith((".", "?")):
        return False                                    # cortada o terminada en otra cosa
    if RE_PUNT_AJENA.search(o):
        return False
    if o.count("¿") != o.count("?"):
        return False                                    # pregunta mal balanceada
    palabras = o.split()
    if len(palabras) < 2:
        return False
    if any(sum(c.isdigit() for c in w) > MAX_DIGITOS_PALABRA for w in palabras):
        return False
    return not re.search(r"https?://|www\.|@", o)


def tramos_validos(turno: str) -> list[list[str]]:
    """Corridas de oraciones válidas consecutivas dentro de un turno.

    Una oración inválida corta la corrida: así un fragmento nunca une dos oraciones que
    en el original no eran contiguas.
    """
    tramos, actual = [], []
    for o in dividir_oraciones(limpiar_turno(turno)):
        if oracion_valida(o):
            actual.append(o)
        elif actual:
            tramos.append(actual)
            actual = []
    if actual:
        tramos.append(actual)
    return tramos


def armar_fragmentos(tramos: list[list[str]], rng: random.Random) -> list[str]:
    """Parte cada tramo en fragmentos consecutivos de 1 a 3 oraciones (largo al azar).

    Como el largo es aleatorio, los fragmentos no empiezan siempre en una pregunta.
    """
    fragmentos = []
    for tramo in tramos:
        i = 0
        while i < len(tramo):
            k = rng.randint(1, MAX_ORACIONES)
            fragmentos.append(" ".join(tramo[i:i + k]))
            i += k
    return fragmentos


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-conferencias", type=int, default=20,
                    help="cuántas conferencias CON periodistas tomar (0 = todas)")
    ap.add_argument("--n-fragmentos", type=int, default=0,
                    help="máximo de fragmentos a guardar, al azar (0 = todos)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default=str(DATASETS / "fragmentos_conferencias.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    urls = listar_conferencias()
    print(f"{len(urls)} conferencias en el listado")
    rng.shuffle(urls)

    filas, usadas = [], 0
    for url in urls:
        if args.n_conferencias and usadas >= args.n_conferencias:
            break
        turnos = extraer_turnos(descargar(url))
        if not tiene_periodistas(turnos):
            print(f"  sin periodistas: {url.split('/')[-1][:80]}")
            continue
        usadas += 1
        tramos = [t for _, texto in turnos for t in tramos_validos(texto)]
        frags = armar_fragmentos(tramos, rng)
        print(f"[{usadas}] {len(turnos)} turnos -> {len(frags)} fragmentos | {url.split('/')[-1][:70]}")
        filas.extend({"Fragmento": f, "Conferencia": url.split("/")[-1]} for f in frags)

    df = pd.DataFrame(filas)
    if args.n_fragmentos and len(df) > args.n_fragmentos:
        df = df.sample(args.n_fragmentos, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "Indice", range(INDICE_INICIAL, INDICE_INICIAL + len(df)))
    df.to_csv(args.salida, index=False)
    print(f"{usadas} conferencias, {len(df)} fragmentos -> {args.salida}")


if __name__ == "__main__":
    main()
