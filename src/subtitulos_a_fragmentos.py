"""
Arma fragmentos (1 a 3 oraciones) a partir de los subtítulos .srt de datasets/subtitulos/subs/.

Reusa las reglas de scrapear_conferencias.py (limpiar_turno, dividir_oraciones,
oracion_valida, tramos_validos, armar_fragmentos) para que los fragmentos salgan igual
que los de las conferencias.

Pipeline:
  1. leer_srt(): bloques (inicio, fin, líneas) sin etiquetas HTML/ASS. limpiar_bloque() saca las
     acotaciones para sordos ([música], (ríe), [Julio]), los carteles y créditos en mayúsculas y las
     líneas de canciones (♪). Cada una queda como None, que corta el turno.
  2. armar_turnos(): une bloques consecutivos en turnos de habla, porque en un .srt las oraciones
     están partidas entre bloques. Un turno nuevo empieza con un guion de diálogo (-), con una
     pausa de más de PAUSA_MS entre dos bloques (corte de escena) o en un None. Al juntar una línea
     con la anterior, decidir_union() corrige la mayúscula que algunos subtitulistas ponen al
     empezar cada bloque ("vamos a jugar a la pelota" / "Y después a comer").
  3. tramos_validos() / armar_fragmentos() del scraper.

Salida: fragmentos_subtitulos.csv (columnas Indice, Fragmento, Conferencia), mismo formato
que fragmentos.csv. `Conferencia` es la película, para poder partir train/test por película.
Después se pasa por armar_instancias.armar_dataset(ruta, desde=0, filtrar=False).
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd

from rutas import DATASETS, SUBTITULOS
from scrapear_conferencias import (armar_fragmentos, dividir_oraciones, limpiar_turno,
                                   oracion_valida, tramos_validos)

INDICE_INICIAL = 200001     # lejos de los Indice de Paenza (1..~3400) y de las conferencias (100001..)
PAUSA_MS = 1500             # silencio entre bloques que corta el turno

RE_TIEMPO = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
RE_ETIQUETAS = re.compile(r"<[^>]+>|\{\\[^}]*\}")       # <i>...</i>, {\an8}
RE_NOTAS = re.compile(r"[♪♫]")
RE_GUION = re.compile(r"^[-–—]\s*")

# Acotaciones para sordos: [música], (ríe), [Julio]. Pueden partirse en dos líneas del mismo
# bloque ("[Golpes en la puerta." / "Ladrido]") o quedar sin cerrar hasta el final del bloque.
RE_ACOTACION = re.compile(r"\[[^\]]*\]|\([^)]*\)")
RE_ACOTACION_ABIERTA = re.compile(r"[\[(][^\])]*$")
RE_CIERRE_HUERFANO = re.compile(r"^[^\[\]]*\]")         # "ROBLES] Hola": el OCR perdió el "[" de apertura
MARCA = "\x02"                                          # ocupa el lugar de una acotación sacada
RE_LETRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
# Dos o más palabras seguidas TODO EN MAYÚSCULAS: créditos como "CINE ACCESIBLE ... FUNDACIÓN ORANGE".
# Una sigla suelta (ESMA, DNI) en medio de una frase no la dispara.
RE_RUN_MAYUS = re.compile(r"(?<!\w)[A-ZÁÉÍÓÚÑÜ]{3,}(?:\s+[A-ZÁÉÍÓÚÑÜ]{3,})+(?!\w)")


def a_ms(h, m, s, ms):
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0")[:3])


def es_cartel(linea: str) -> bool:
    """Línea entera en mayúsculas (INTERVALO, patentes) o con 2+ palabras seguidas en mayúsculas."""
    s = RE_GUION.sub("", linea)
    letras = [c for c in s if c.isalpha()]
    if len(letras) >= 2 and s == s.upper():
        return re.sub(r"\W", "", s) != "OK"             # "OK" se dice y es clase 3 de verdad
    return bool(RE_RUN_MAYUS.search(s))


def limpiar_bloque(lineas: list[str]) -> list[str | None]:
    """Saca de las líneas de un bloque las acotaciones, los carteles y las líneas de canciones.

    Devuelve las líneas de habla; None marca un corte de turno: la línea era solo una acotación, un
    cartel o una canción, o arrancaba con una acotación ("[Julio] ¿Qué pasa?" suele ser un cambio
    de hablante). Se hace por línea, antes de unirlas, para que una acotación entre dos líneas de
    una misma frase no tape si la anterior quedó abierta.
    """
    texto = RE_ACOTACION.sub(MARCA, "\n".join(lineas))
    texto = RE_ACOTACION_ABIERTA.sub(MARCA, texto)
    salida = []
    for linea in texto.split("\n"):
        linea = RE_CIERRE_HUERFANO.sub(MARCA, linea)
        habla = linea.replace(MARCA, "").strip()
        if not RE_LETRA.search(habla) or RE_NOTAS.search(habla) or es_cartel(habla):
            salida.append(None)
            continue
        if linea.lstrip().startswith(MARCA):
            salida.append(None)
        salida.append(bajar_pregunta_tras_coma(habla))
    return salida


def leer_srt(ruta: Path) -> list[tuple[int, int, list[str | None]]]:
    """Lista de (inicio_ms, fin_ms, líneas de habla) de un .srt; ver limpiar_bloque() por los None."""
    texto = ruta.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    bloques = []
    for crudo in re.split(r"\n\s*\n", texto):
        lineas = crudo.strip().split("\n")
        i = next((k for k, l in enumerate(lineas) if RE_TIEMPO.search(l)), None)
        if i is None:
            continue
        g = RE_TIEMPO.search(lineas[i]).groups()
        contenido = [RE_ETIQUETAS.sub("", l).strip() for l in lineas[i + 1:]]
        contenido = [l for l in contenido if l]
        if contenido:
            bloques.append((a_ms(*g[:4]), a_ms(*g[4:]), limpiar_bloque(contenido)))
    return bloques


# ---------------------------------------------------------------------------
# Unión de líneas: mayúsculas espurias y frases nuevas sin cerrar
# ---------------------------------------------------------------------------
RE_CIERRA = re.compile(r"[.?!…][\"'”»)\]]*$")           # la línea termina una oración
RE_INICIO = re.compile(r"^¿?([A-ZÁÉÍÓÚÑ][a-záéíóúüñ]*)(?![^\W\d_])")   # "Y", "Pero", "¿Qué"; no "NO"
RE_PRIMERA = re.compile(r"^\W*([^\W\d_]+)")

# Palabras con las que una línea no puede terminar una frase: lo que sigue la continúa.
FIN_ABIERTO = set(
    "y o u e ni pero sino que porque pues aunque mientras cuando como si donde "
    "a al de del en con sin por para hasta desde sobre entre contra según ante bajo tras "
    "el la los las un una unos unas mi tu su mis tus sus lo le les me te se nos".split()
)
# Palabras función que no se escriben con mayúscula en medio de una frase (y que no son nombres propios).
FUNCION = FIN_ABIERTO | set(
    "no ya así también tampoco yo vos tú él ella entonces "
    "qué cómo dónde cuándo cuál cuáles quién quiénes cuánto cuánta cuántos cuántas".split()
)
# Pueden formar parte de un nombre propio si sigue otra palabra en mayúscula: La Plata, De Agostini.
PARTICULAS_NOMBRE = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del"}


def estado_previa(previa: str) -> str:
    """'cerrada' (termina una oración), 'dos_puntos' (lo que sigue es una cita o enumeración),
    'continua' (termina en coma o en una palabra que no cierra frase) o 'dudosa' (no se sabe)."""
    previa = previa.rstrip()
    if RE_CIERRA.search(previa):
        return "cerrada"
    if previa.endswith(":"):
        return "dos_puntos"
    if previa.endswith((",", ";")):
        return "continua"
    ultima = re.sub(r"\W+$", "", previa.split()[-1])
    return "continua" if ultima.islower() and ultima in FIN_ABIERTO else "dudosa"   # "grado A" no es la preposición


def se_baja(palabra: str, resto: str) -> bool:
    """¿Se puede escribir `palabra` (seguida de `resto`) con minúscula sin perder un nombre propio?"""
    bajo = palabra.lower()
    if bajo not in FUNCION:
        return False
    if len(palabra) == 1 and resto[:1] == ".":          # inicial de un nombre: "E. Sábato"
        return False
    return not (bajo in PARTICULAS_NOMBRE and resto.lstrip()[:1].isupper())


RE_PREGUNTA_TRAS_COMA = re.compile(r"(,\s*¿)([A-ZÁÉÍÓÚÑ][a-záéíóúüñ]*)(?![^\W\d_])")


def bajar_pregunta_tras_coma(linea: str) -> str:
    """'Mi hija, ¿Dónde está?' -> 'Mi hija, ¿dónde está?': tras una coma la pregunta sigue la frase."""
    return RE_PREGUNTA_TRAS_COMA.sub(
        lambda m: m.group(1) + (m.group(2).lower() if se_baja(m.group(2), linea[m.end(2):]) else m.group(2)),
        linea)


def decidir_union(previa: str, linea: str) -> tuple[str, str]:
    """Qué hacer al juntar `linea` con la línea anterior del mismo turno.

    Devuelve (acción, línea): 'seguir' (nada), 'minuscula' (se baja la mayúscula de la primera
    palabra), 'cortar' (frase nueva: se cierra el turno) o 'ambigua' (mayúscula tras una línea
    abierta que no se toca: nombre propio, vocativo, verso de una canción o frase nueva).
    """
    estado = estado_previa(previa)
    if estado in ("cerrada", "dos_puntos"):
        return "seguir", linea
    if estado == "dudosa" and linea.startswith("¿"):
        return "cortar", linea              # pregunta nueva sobre una línea que quedó sin punto
    if estado == "continua":
        m = RE_INICIO.match(linea)
        if m and se_baja(m.group(1), linea[m.end(1):]):
            return "minuscula", linea[:m.start(1)] + m.group(1).lower() + linea[m.end(1):]
    p = RE_PRIMERA.match(linea)
    if p and p.group(1)[0].isupper() and not (len(p.group(1)) > 1 and p.group(1).isupper()):
        return "ambigua", linea
    return "seguir", linea


def armar_turnos(bloques: list[tuple[int, int, list[str | None]]],
                 eventos: list | None = None) -> list[str]:
    """Une las líneas de habla en turnos. Si se pasa `eventos`, se le agrega (acción, línea previa,
    línea) de cada unión que no fue 'seguir', para revisar las decisiones."""
    turnos, actual, fin_previo = [], [], None

    def cerrar():
        nonlocal actual
        if actual:
            turnos.append(" ".join(actual))
        actual = []

    for inicio, fin, lineas in bloques:
        if fin_previo is not None and inicio - fin_previo > PAUSA_MS:
            cerrar()
        for linea in lineas:
            if linea is None:
                cerrar()
            elif RE_GUION.match(linea):
                cerrar()
                actual = [RE_GUION.sub("", linea)]
            else:
                if actual:
                    accion, nueva = decidir_union(actual[-1], linea)
                    if eventos is not None and accion != "seguir":
                        eventos.append((accion, actual[-1], nueva))
                    if accion == "cortar":
                        cerrar()
                    linea = nueva
                actual.append(linea)
        fin_previo = fin
    cerrar()
    return turnos


def nombre_pelicula(archivo: Path) -> str:
    """'La ciénaga (2001) [ea-11592616].srt' -> 'La ciénaga (2001)'."""
    return re.sub(r"\s*\[[^\]]*\]$", "", archivo.stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", nargs="?", default=str(SUBTITULOS / "subs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default=str(DATASETS / "fragmentos_subtitulos.csv"))
    args = ap.parse_args()

    filas = []
    for archivo in sorted(Path(args.carpeta).glob("*.srt")):
        # Un sorteo por película: así sus fragmentos no cambian al sumar o sacar otras películas.
        rng = random.Random(f"{args.seed}-{nombre_pelicula(archivo)}")
        bloques = leer_srt(archivo)
        turnos = armar_turnos(bloques)
        oraciones = [o for t in turnos for o in dividir_oraciones(limpiar_turno(t))]
        validas = sum(oracion_valida(o) for o in oraciones)
        tramos = [tr for t in turnos for tr in tramos_validos(t)]
        frags = armar_fragmentos(tramos, rng)
        print(f"{nombre_pelicula(archivo)}: {len(bloques)} bloques -> {len(turnos)} turnos -> "
              f"{validas}/{len(oraciones)} oraciones válidas -> {len(frags)} fragmentos")
        filas.extend({"Fragmento": f, "Conferencia": nombre_pelicula(archivo)} for f in frags)

    df = pd.DataFrame(filas)
    df.insert(0, "Indice", range(INDICE_INICIAL, INDICE_INICIAL + len(df)))
    df.to_csv(args.salida, index=False)
    palabras = df["Fragmento"].str.split().str.len().sum()
    print(f"{len(df)} fragmentos, {palabras} palabras -> {args.salida}")


if __name__ == "__main__":
    main()
