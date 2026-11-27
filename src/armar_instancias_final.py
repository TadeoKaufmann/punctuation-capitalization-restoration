"""
Arma datasets/instancias_final/instancias_final.csv: instancias de texto (con su puntuación y mayúsculas
originales) divididas en 4 clases, en todas las fuentes del TP: Paenza, conferencias, subtítulos, el
dataset sintético y una selección de Wikinoticias (ver armar_wikinoticias.py y seleccionar_wikinoticias.py). El enunciado dice que una instancia "puede contener parte de una o varias oraciones".

Clases:
  parte        un pedazo de UNA oración: la oración se parte ENTERA en 2 a 5 pedazos consecutivos
               (inicio, medio(s), fin; >= 3 palabras cada uno), así no se pierde ninguna palabra
  1_oracion    una oración completa
  2_oraciones  dos oraciones consecutivas
  3_o_mas      de 3 a 6 oraciones consecutivas

Cómo se arma: cada fuente se lee como tramos de oraciones contiguas (nunca se pegan oraciones que en el
original no eran vecinas). Cada tramo se recorre en orden y cada oración se usa UNA sola vez: en cada paso
se sortea la clase entre las posibles, con un refuerzo suave hacia la clase que va más atrasada respecto del
reparto objetivo (25 % de las instancias para cada clase). Un sorteo por grupo
(película, conferencia, ...), sembrado con el nombre del grupo, para que sumar o sacar un grupo no cambie al
resto (mismo criterio que subtitulos_a_fragmentos.py).

Uso (desde cualquier carpeta):
  python src/armar_instancias_final.py
"""
import argparse
import gzip
import json
import pickle
import random
import re
from collections import Counter

import pandas as pd

from armar_instancias import INDICE_DESDE, filtrar_fragmentos
from rutas import DATASETS, SALIDAS, SUBTITULOS
from scrapear_conferencias import dividir_oraciones, tramos_validos
from subtitulos_a_fragmentos import armar_turnos, leer_srt, nombre_pelicula

SALIDA = DATASETS / "instancias_final"
CLASES = ["parte", "1_oracion", "2_oraciones", "3_o_mas"]
OBJETIVO = {c: 1 / len(CLASES) for c in CLASES}
PARTES = ["inicio", "medio", "fin"]

MIN_PARTE = 3                    # palabras de un pedazo de oración (una oración de menos de 2 * MIN_PARTE no se parte)
MAX_PARTES = 5                   # en cuántos pedazos como máximo se parte una oración
MAX_ORACIONES = 6                # tope de la clase 3_o_mas
MIN_PAL_ORACION = 2              # como oracion_valida() de scrapear_conferencias.py
MAX_PAL_ORACION = 30             # una oración más larga no entra entera: se parte en pedazos (clase parte), ver achicar_oracion()
MIN_PEDAZO_LARGA = 5             # palabras mínimas de un pedazo de una oración larga
MAX_PAL_INSTANCIA = 150

# Indice único en todo el archivo. Paenza arranca en 101 para pasar el filtro Indice >= 100 del loader actual.
BASES = {"paenza": 101, "conferencias": 100001, "subtitulos": 200001, "sintetico": 300001, "wikinoticias": 400001}
FUENTES = list(BASES)
ARMADAS = [f for f in FUENTES if f != "wikinoticias"]   # las que arma este script; wikinoticias llega ya armada
# Las 3 oraciones por fragmento de Paenza salen de la regex del Generador_de_instancias de la cátedra
RE_ORACION_CATEDRA = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")
RE_AJENA = re.compile(r"[…:;!¡]|\.\.")


# ---------------------------------------------------------------------------
# Lectura: cada fuente -> {grupo: [tramo, ...]}, tramo = [(oración, origen), ...] contiguas
# ---------------------------------------------------------------------------
def cortar_por_largo(oraciones: list[tuple[str, str]], descartadas: Counter, fuente: str, invalida=None) -> list[list]:
    """Parte la lista en corridas. Una oración de menos de MIN_PAL_ORACION palabras, o que `invalida(texto)`
    rechaza (devuelve el motivo), se descarta y corta la corrida."""
    tramos, actual = [], []
    for texto, origen in oraciones:
        motivo = "corta" if len(texto.split()) < MIN_PAL_ORACION else (invalida(texto) if invalida else None)
        if motivo is None:
            actual.append((texto, origen))
        else:
            descartadas[fuente] += 1
            descartadas[f"{fuente}:{motivo}"] += 1
            if actual:
                tramos.append(actual)
                actual = []
    if actual:
        tramos.append(actual)
    return tramos


# Ruido del PDF de Paenza (texto de dos columnas mezclado, notas al pie y títulos pegados en medio de una oración)
RE_PALABRA = re.compile(r"[a-záéíóúüñ]+")
RE_NOTA_PEGADA = re.compile(r"\d[A-ZÁÉÍÓÚÑ][a-záéíóúñ]|[A-Za-záéíóúñ]{3,}\d+(?![\d%.,])")   # "12El", "cola.27Por"
RE_PEGADO = re.compile(r"[a-záéíóúñ][A-ZÁÉÍÓÚÑ]")                                              # "toLuego", "balVamos"
RE_CARA_CECA = re.compile(r"(?:Cara|Ceca)+")                                                 # notación legítima (monedas)
RE_TITULO = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ]{3,}\s+[A-ZÁÉÍÓÚÑ]{3,}\b"                                                  # "A) SOBRE LOS BARBEROS"
    r"|(?<![A-Za-z])(?:SOLUCI[ÓO]N(?:ES)?|PROBLEMAS?|MORALEJA|EJEMPLOS?|AP[ÉE]NDICE|MISCEL[ÁA]NEA|OBSERVACIONES)(?![a-záéíóúñ])")
BORDES = ".,;:?¿¡!()\"“”«»-–—"


def vocabulario_europarl() -> set:
    """Palabras de Europarl (>= 3 apariciones): diccionario de español para detectar palabras rotas del PDF."""
    cache = SALIDAS / "vocab_europarl.pkl"
    if cache.exists():
        return set(pickle.loads(cache.read_bytes()))
    ruta = DATASETS / "no-argentinos" / "europarl_es_v7" / "es.txt.gz"
    if not ruta.exists():
        print(f"Aviso: falta {ruta}; no se detectan palabras desconocidas en Paenza.")
        return set()
    cuentas: Counter = Counter()
    with gzip.open(ruta, "rt", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            cuentas.update(RE_PALABRA.findall(linea.lower()))
    vocab = {w for w, n in cuentas.items() if n >= 3}
    cache.write_bytes(pickle.dumps(vocab))
    return vocab


def ruido_paenza(texto: str, vocab: set) -> str | None:
    """Motivo por el que una oración de Paenza trae ruido del PDF, o None si está limpia."""
    palabras = [w.strip(BORDES) for w in texto.split()]
    if any(RE_NOTA_PEGADA.search(w) for w in palabras):
        return "nota al pie pegada"
    if any(RE_PEGADO.search(w) and not RE_CARA_CECA.fullmatch(w) for w in palabras):
        return "palabras pegadas"
    if RE_TITULO.search(texto):
        return "título en mayúsculas"
    if vocab and any(re.fullmatch(r"[a-záéíóúüñ]{4,}", w) and w not in vocab for w in palabras):
        return "palabra desconocida"
    return None


def leer_paenza(descartadas: Counter) -> dict:
    df = pd.read_csv(DATASETS / "fragmentos.csv")
    df = df[df["Indice"] >= INDICE_DESDE].dropna(subset=["Fragmento"])
    df = filtrar_fragmentos(df)
    propio = Counter(w for t in df["Fragmento"] for w in RE_PALABRA.findall(t.lower()))
    vocab = vocabulario_europarl() | {w for w, n in propio.items() if n >= 2}   # una palabra que Paenza usa 2+ veces es real
    # fragmentos.csv está en el orden del libro: Indice consecutivos son texto contiguo; un hueco (fragmento
    # que filtrar_fragmentos descartó) corta la corrida.
    corridas, actual, previo = [], [], None
    for indice, texto in zip(df["Indice"], df["Fragmento"]):
        if previo is not None and indice != previo + 1:
            corridas.append(actual)
            actual = []
        actual += [(o, f"paenza:{indice}") for o in RE_ORACION_CATEDRA.split(texto) if o.strip()]
        previo = indice
    corridas.append(actual)
    invalida = lambda o: ruido_paenza(o, vocab)
    return {"paenza": [t for c in corridas for t in cortar_por_largo(c, descartadas, "paenza", invalida)]}


def leer_conferencias(descartadas: Counter) -> dict:
    # fragmentos_conferencias.csv está mezclado al azar: solo se sabe que las oraciones DE UN fragmento son contiguas
    df = pd.read_csv(DATASETS / "fragmentos_conferencias.csv").dropna(subset=["Fragmento"])
    grupos: dict[str, list] = {}
    for indice, texto, conf in zip(df["Indice"], df["Fragmento"], df["Conferencia"]):
        oraciones = [(o, f"conf:{indice}") for o in dividir_oraciones(texto)]
        grupos.setdefault(conf, []).extend(cortar_por_largo(oraciones, descartadas, "conferencias"))
    return grupos


def leer_subtitulos(descartadas: Counter) -> dict:
    # Mismo pipeline que subtitulos_a_fragmentos.main(), pero quedándose con los tramos enteros
    grupos = {}
    for archivo in sorted((SUBTITULOS / "subs").glob("*.srt")):
        pelicula = nombre_pelicula(archivo)
        tramos = []
        for turno in armar_turnos(leer_srt(archivo)):
            for tramo in tramos_validos(turno):
                k = len(tramos) + 1
                oraciones = [(o, f"sub:{pelicula}#{k}.{j}") for j, o in enumerate(tramo, 1)]
                tramos.extend(cortar_por_largo(oraciones, descartadas, "subtitulos"))
        grupos[pelicula] = tramos
    return grupos


def leer_wikinoticias() -> pd.DataFrame:
    """Instancias ya armadas por armar_wikinoticias.py + seleccionar_wikinoticias.py (se copian tal cual)."""
    return pd.read_csv(DATASETS / "wikinoticias" / "instancias_wikinoticias_sel.csv", keep_default_na=False)


def leer_sintetico(descartadas: Counter) -> dict:
    ruta = DATASETS / "dataset_sintetico" / "dataset_sintetico.jsonl"
    tramos = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            reg = json.loads(linea)
            oraciones = [(o, reg["id"]) for o in dividir_oraciones(reg["gold"])]
            tramos.extend(cortar_por_largo(oraciones, descartadas, "sintetico"))
    return {"sintetico": tramos}


# ---------------------------------------------------------------------------
# Armado de instancias
# ---------------------------------------------------------------------------
def elegir(rng: random.Random, factibles: list[str], objetivo: dict, cuentas: Counter, filas_por_paso: dict) -> str:
    """Sortea entre las opciones posibles; pesa más a la que va más atrasada respecto del objetivo.

    `filas_por_paso[c]`: cuántas instancias da en promedio elegir la clase c (una oración partida da varias):
    se divide el peso por eso para que el reparto objetivo sea de instancias y no de decisiones.
    """
    total = sum(cuentas[c] for c in objetivo)
    pesos = []
    for c in factibles:
        actual = cuentas[c] / total if total else 0.0
        pesos.append(objetivo[c] * max(0.1, 1 + 8 * (objetivo[c] - actual) / objetivo[c]) / filas_por_paso.get(c, 1.0))
    return rng.choices(factibles, weights=pesos)[0]


def partir_oracion(palabras: list[str], rng: random.Random) -> list[str]:
    """Parte TODA la oración (de hasta MAX_PAL_ORACION palabras) en k pedazos consecutivos de >= MIN_PARTE
    palabras, con los cortes al azar. Requiere len(palabras) >= 2 * MIN_PARTE."""
    n = len(palabras)
    k = rng.randint(2, min(MAX_PARTES, n // MIN_PARTE))
    extra = n - MIN_PARTE * k                   # palabras a repartir entre los k pedazos, además del mínimo
    cortes = sorted(rng.sample(range(extra + k - 1), k - 1))
    limites = [-1] + cortes + [extra + k - 1]
    tamanos = [MIN_PARTE + b - a - 1 for a, b in zip(limites, limites[1:])]
    assert sum(tamanos) == n and min(tamanos) >= MIN_PARTE
    pedazos, desde = [], 0
    for tam in tamanos:
        pedazos.append(" ".join(palabras[desde:desde + tam]))
        desde += tam
    assert " ".join(pedazos) == " ".join(palabras)
    return pedazos


# Antes de estas palabras empieza una cláusula nueva: "..., y que ...", "... pero ...", "... porque ..."
CONECTORES = {"y", "e", "o", "u", "ni", "pero", "sino", "que", "porque", "aunque", "cuando", "mientras", "donde",
              "como", "si", "pues"}
COSTO_CORTE = {"coma": 0.0, "conector": 3.0, "debil": 12.0}


def tipo_corte(palabras: list[str], p: int) -> str:
    """Cómo sería cortar entre palabras[p-1] y palabras[p]: 'coma' (la anterior cierra con , ; : ?),
    'conector' (la siguiente abre una cláusula) o 'debil' (cualquier otro lugar)."""
    if palabras[p - 1][-1:] in {",", ";", ":", "?"}:
        return "coma"
    if palabras[p].lower().lstrip("¿¡\"'(“«-–—") in CONECTORES:
        return "conector"
    return "debil"


def achicar_oracion(palabras: list[str], maximo: int = MAX_PAL_ORACION, minimo: int = MIN_PEDAZO_LARGA):
    """Parte una oración de más de `maximo` palabras en pedazos consecutivos de `minimo` a `maximo` palabras.

    No corta cada `maximo` palabras: elige los cortes donde la oración tiene sentido (después de una coma, antes
    de un conector) y, entre esos, los que dejan pedazos parejos. Solo si no hay ninguno posible corta en
    cualquier lado ('debil'). Prueba con la cantidad mínima de pedazos y con uno más, y se queda con lo más lógico.
    Devuelve (pedazos, tipos de corte).
    """
    n = len(palabras)
    k0 = -(-n // maximo)
    opciones = []
    for k in (k0, k0 + 1):
        if k * minimo > n:
            continue
        ideal = n / k
        estados = [{0: (0.0, None)}]            # estados[j][p] = (costo mínimo, posición del corte anterior)
        for j in range(1, k):
            nuevo = {}
            for p in range(minimo * j, n - minimo * (k - j) + 1):
                costo_p = COSTO_CORTE[tipo_corte(palabras, p)] + 0.5 * abs(p - j * ideal)
                previos = [(c + costo_p, q) for q, (c, _) in estados[j - 1].items() if minimo <= p - q <= maximo]
                if previos:
                    nuevo[p] = min(previos)
            estados.append(nuevo)
        finales = [(estados[k - 1][p][0], p) for p in estados[k - 1] if minimo <= n - p <= maximo]
        if finales:
            costo, p = min(finales)
            opciones.append((costo + 4 * (k - k0), k, estados, p))
    _, k, estados, p = min(opciones, key=lambda o: o[0])
    cortes = [p]
    for j in range(k - 1, 1, -1):
        cortes.append(estados[j][cortes[-1]][1])
    cortes = cortes[::-1]
    limites = [0] + cortes + [n]
    pedazos = [" ".join(palabras[a:b]) for a, b in zip(limites, limites[1:])]
    assert " ".join(pedazos) == " ".join(palabras) and all(minimo <= len(x.split()) <= maximo for x in pedazos)
    return pedazos, [tipo_corte(palabras, c) for c in cortes]


def agregar_pedazos(instancias: list[dict], pedazos: list[str], origen: str) -> int:
    """Agrega los pedazos de UNA oración como instancias de clase parte (inicio, medios, fin); devuelve cuántos."""
    primera = len(instancias)
    nombres = ["inicio"] + ["medio"] * (len(pedazos) - 2) + ["fin"]
    for texto, nombre in zip(pedazos, nombres):
        instancias.append({
            "Fragmento": texto, "clase": "parte", "parte": nombre, "n_oraciones": 0,
            "n_palabras": len(texto.split()), "origen": origen, "_primera": primera,
        })
    return len(pedazos)


def armar_grupo(tramos: list[list], rng: random.Random, stats: Counter, fuente: str) -> list[dict]:
    instancias, cuentas, pasos_parte = [], Counter(), 0
    for tramo in tramos:
        i = 0
        while i < len(tramo):
            palabras0 = tramo[i][0].split()
            if len(palabras0) > MAX_PAL_ORACION:         # oración larga: siempre en pedazos, cortes lógicos
                pedazos, tipos = achicar_oracion(palabras0)
                cuentas["parte"] += agregar_pedazos(instancias, pedazos, tramo[i][1])
                stats[f"{fuente}:achicadas"] += 1
                stats.update(f"{fuente}:corte {t}" for t in tipos)
                i += 1
                continue

            ventana = [t for t, _ in tramo[i:i + MAX_ORACIONES]]
            pal = [len(t.split()) for t in ventana]
            cortas = next((j for j, p in enumerate(pal) if p > MAX_PAL_ORACION), len(pal))
            ventana, pal = ventana[:cortas], pal[:cortas]          # las de 2 y 3+ solo juntan oraciones de <= 30
            factibles = ["1_oracion"]
            if pal[0] >= 2 * MIN_PARTE:
                factibles.append("parte")
            if len(ventana) >= 2 and sum(pal[:2]) <= MAX_PAL_INSTANCIA:
                factibles.append("2_oraciones")
            if len(ventana) >= 3 and sum(pal[:3]) <= MAX_PAL_INSTANCIA:
                factibles.append("3_o_mas")

            filas_por_paso = {"parte": cuentas["parte"] / pasos_parte if pasos_parte else 3.0}
            clase = elegir(rng, factibles, OBJETIVO, cuentas, filas_por_paso)
            if clase == "parte":
                pasos_parte += 1
                cuentas[clase] += agregar_pedazos(instancias, partir_oracion(ventana[0].split(), rng), tramo[i][1])
                i += 1
                continue

            usadas = 1
            if clase == "1_oracion":
                texto = ventana[0]
            elif clase == "2_oraciones":
                texto, usadas = " ".join(ventana[:2]), 2
            else:
                k_max = max(k for k in range(3, len(ventana) + 1) if sum(pal[:k]) <= MAX_PAL_INSTANCIA)
                usadas = rng.randint(3, k_max)
                texto = " ".join(ventana[:usadas])

            texto = " ".join(texto.split())
            cuentas[clase] += 1
            instancias.append({
                "Fragmento": texto, "clase": clase, "parte": "", "n_oraciones": usadas,
                "n_palabras": len(texto.split()), "origen": tramo[i][1], "_primera": len(instancias),
            })
            i += usadas
    return instancias


def armar_todo(semilla: int) -> tuple[pd.DataFrame, Counter]:
    descartadas: Counter = Counter()
    lectores = {"paenza": leer_paenza, "conferencias": leer_conferencias,
                "subtitulos": leer_subtitulos, "sintetico": leer_sintetico}
    filas = []
    for fuente in ARMADAS:
        grupos = lectores[fuente](descartadas)
        n_fuente = 0
        for grupo, tramos in grupos.items():
            rng = random.Random(f"{semilla}-{fuente}-{grupo}")
            instancias = armar_grupo(tramos, rng, descartadas, fuente)
            for j, inst in enumerate(instancias):
                bloque = BASES[fuente] + n_fuente + inst.pop("_primera")
                filas.append({"Indice": BASES[fuente] + n_fuente + j, "documento": grupo, "fuente": fuente,
                              "bloque": bloque, **inst})
            n_fuente += len(instancias)
    df = pd.DataFrame(filas)
    columnas = ["Indice", "Fragmento", "documento", "fuente", "clase", "parte", "n_oraciones", "n_palabras", "origen", "bloque"]
    wiki = leer_wikinoticias()
    return pd.concat([df[columnas], wiki[columnas]], ignore_index=True), descartadas


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
def tabla_md(df: pd.DataFrame) -> str:
    filas = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    filas += ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(filas)


def resumen_clases(df: pd.DataFrame) -> pd.DataFrame:
    cuentas = pd.crosstab(df["fuente"], df["clase"]).reindex(index=FUENTES, columns=CLASES, fill_value=0)
    cuentas["total"] = cuentas.sum(axis=1)
    total = cuentas.sum().rename("TOTAL").to_frame().T
    tabla = pd.concat([cuentas, total])
    pct = (tabla[CLASES].div(tabla["total"], axis=0) * 100).round(1)
    out = tabla.astype(str)
    for c in CLASES:
        out[c] = tabla[c].astype(str) + " (" + pct[c].astype(str) + " %)"
    return out.reset_index(names="fuente")


def resumen_palabras(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("fuente")["n_palabras"].agg(["sum", "mean", "max"]).reindex(FUENTES)
    g["mean"] = g["mean"].round(1)
    g.columns = ["palabras", "palabras por instancia (media)", "palabras por instancia (máx)"]
    return g.reset_index()


def escribir_readme(df: pd.DataFrame, descartadas: Counter, semilla: int) -> None:
    partes = pd.crosstab(df["fuente"], df["parte"].replace("", pd.NA).dropna()).reindex(index=FUENTES, columns=PARTES, fill_value=0)
    pct_signos = round(100 * df.loc[df["fuente"] == "paenza", "Fragmento"].str.contains(r"[¡!:;…]").mean())
    achicadas = ", ".join(f"{k} {descartadas[f'{k}:achicadas']}" for k in ARMADAS)
    cortes = {t: sum(descartadas[f"{k}:corte {t}"] for k in ARMADAS) for t in COSTO_CORTE}
    motivos = ["nota al pie pegada", "palabras pegadas", "título en mayúsculas", "palabra desconocida"]
    ruido = ", ".join(f"{m} {descartadas[f'paenza:{m}']}" for m in motivos)
    texto = f"""# instancias_final — instancias de texto en 4 clases

Generado por `src/armar_instancias_final.py` (semilla {semilla}); no editar a mano. Las instancias son texto con su
puntuación y mayúsculas originales; las etiquetas por palabra y por token las saca el pipeline (`palabras_y_etiquetas`,
`armar_instancias.etiquetar_texto`), como con `fragmentos_todos.csv`.

El enunciado dice que una instancia "puede contener parte de una o varias oraciones". Se divide en 4 clases, en **todas** las fuentes:

| clase | qué es |
|---|---|
| `parte` | un pedazo de **una** oración. La oración se parte **entera** en pedazos consecutivos (`inicio`, `medio`s, `fin`): cada oración partida da tantas instancias como pedazos y no se pierde ninguna palabra. Se parten al azar (2 a {MAX_PARTES} pedazos de >= {MIN_PARTE} palabras) las oraciones que salen sorteadas como `parte`, y siempre en cortes lógicos las de más de {MAX_PAL_ORACION} palabras |
| `1_oracion` | una oración completa |
| `2_oraciones` | dos oraciones consecutivas |
| `3_o_mas` | de 3 a {MAX_ORACIONES} oraciones consecutivas |

## Cuántas hay

{tabla_md(resumen_clases(df))}

Partes de oración (inicio / medio / fin):

{tabla_md(partes.reset_index())}

{tabla_md(resumen_palabras(df))}

## Cómo se armó

- Cada fuente se lee como **tramos de oraciones contiguas** y no se pegan oraciones que en el original no eran vecinas.
- Cada oración del original se usa **una sola vez**: ninguna aparece en dos instancias. (Pueden repetirse textos cortos que el original ya dice varias veces, como "¿Qué?" en los subtítulos.)
- En cada paso se sortea la clase entre las posibles, con un refuerzo suave hacia la clase más atrasada (objetivo: 25 % de las instancias para cada clase). Una oración de menos de {2 * MIN_PARTE} palabras no se parte. El resultado depende de cuántas oraciones seguidas hay en cada fuente (tabla de arriba).
- Un pedazo (`parte`) conserva la puntuación y las mayúsculas del original: un `inicio` arranca en mayúscula (y quizá `¿`) sin cerrar, un `fin` arranca en minúscula y cierra con `.` o `?`, un `medio` no tiene ni lo uno ni lo otro. Los pedazos de una misma oración quedan en filas seguidas y comparten `bloque`: si se parte train/test, hay que hacerlo por `documento` o por `bloque`, no por fila, para que no haya fuga.
- **Máximo {MAX_PAL_ORACION} palabras por oración.** Una oración más larga no entra entera en `1_oracion`, `2_oraciones` ni `3_o_mas`: se parte en pedazos de {MIN_PEDAZO_LARGA} a {MAX_PAL_ORACION} palabras (clase `parte`: `inicio`, `medio`s, `fin`) y no se pierde ninguna palabra. No se corta cada {MAX_PAL_ORACION} palabras: `achicar_oracion` busca los cortes donde la oración tiene sentido (después de una coma, `;`, `:` o `?`, o antes de un conector como `que`, `y`, `pero`, `porque`) y, entre esos, los que dejan pedazos parejos; si no hay ninguno posible, corta en cualquier lado. Oraciones achicadas: {achicadas}. Cortes: {cortes["coma"]} después de coma, {cortes["conector"]} antes de un conector, {cortes["debil"]} en cualquier lado.
- Se descartan (y cortan el tramo) las oraciones de menos de {MIN_PAL_ORACION} palabras y, en Paenza, las que traen ruido del PDF: {", ".join(f"{k} {descartadas[k]}" for k in ARMADAS)} en total. Ninguna instancia pasa de {MAX_PAL_INSTANCIA} palabras.

## Qué tiene cada fuente (y sus límites)

- **paenza**: los fragmentos del libro ya filtrados con `filtrar_fragmentos` (cuentas, editorial, encabezados), en orden; un hueco de `Indice` corta el tramo. Las oraciones se separan con la regex del Generador de instancias de la cátedra (separa también en `!`). **No** se les aplicó `oracion_valida`: quedan oraciones con signos que el TP no predice (`¡ ! : ; …`) y esos signos se quedan en el input (decisión de Tadeo, 2026-11-16). Son el {pct_signos} % de las instancias de Paenza; en las otras fuentes no hay, porque se descartaron al armar los fragmentos. Ojo: `normalizar_texto` los conserva, pero `palabras_y_etiquetas` los saca de la palabra que se convierte en embedding. Se limpió el ruido del PDF: se descartan las oraciones con notas al pie pegadas a una palabra (`12El`), palabras pegadas por las columnas (`toLuego`; se exceptúa `CaraCara`/`CecaCeca`), títulos en mayúsculas pegados (`A) SOBRE LOS BARBEROS`, `SOLUCIÓN`) y palabras que no están en un diccionario de español (Europarl, más lo que Paenza usa 2 o más veces). Descartadas: {ruido}. La última regla es la menos precisa: a ojo, 1 de cada 5 palabras marcadas es un término raro real. No se detectan números de página sueltos en medio de una oración.
- **conferencias**: `fragmentos_conferencias.csv` está mezclado al azar, así que solo se sabe que las oraciones **de un mismo fragmento** son contiguas. Por eso `3_o_mas` son exactamente 3 oraciones. Para tener más hay que volver a bajar las conferencias.
- **subtitulos**: se rehicieron los tramos desde los `.srt` con el mismo pipeline de `subtitulos_a_fragmentos.py`, así que los tramos son completos. Las oraciones son cortas (mediana de 5 palabras): muchas no alcanzan para cortar un pedazo.
- **sintetico**: cada línea del `.jsonl` es una instancia independiente de 1 a 4 oraciones; no se pegan líneas entre sí.
- **wikinoticias**: notas de es.wikinews.org de las categorías ONU, Diplomacia, Unión Europea, Mundo, Economía y Negocios, Democracia y elecciones y Política (`src/descargar_wikinoticias.py`), armadas con el mismo criterio (`src/armar_wikinoticias.py`) y **seleccionadas** con `src/seleccionar_wikinoticias.py`: solo entran las oraciones con palabras de capitalización clase 2 (mixta: `WikiLeaks`, `iPhone`) o clase 3 (siglas: `ONU`, `OTAN`), priorizando tipos raros y con tope por tipo. Se agregó **para subir los ejemplos de esas dos clases**, que en las otras fuentes son muy pocos. Por eso **no es representativa**: no refleja la distribución natural de mayúsculas, ni la de puntuación (es registro de noticia, casi sin `?`), y el 70 % son `parte` porque las oraciones de noticia son largas. Se copia tal cual desde `datasets/wikinoticias/instancias_wikinoticias_sel.csv` (no se vuelve a armar acá). Se quitaron comillas y paréntesis, y "EE. UU." quedó como `EEUU`. Las clase 2 incluyen apellidos con guión (`Strauss-Kahn`, `Al-Assad`), que la regla de `clase_capitalizacion` marca como clase 2.
- **No incluidos**: `cuentos_et`, `cuentos_argentinos_recientes` y `no-argentinos/` (Europarl, SQuAD, FLORES, OpenSubtitles general). No están en el pipeline de entrenamiento y los PDF de los cuentos necesitan limpieza.

## Columnas

`Indice` (único; paenza desde {BASES['paenza']}, conferencias desde {BASES['conferencias']}, subtitulos desde {BASES['subtitulos']}, sintetico desde {BASES['sintetico']}, wikinoticias desde {BASES['wikinoticias']}) · `Fragmento` (texto) · `documento` (el documento de donde viene, para partir train/test: la película, la conferencia, la nota de Wikinoticias, `paenza` o `sintetico`; antes se llamaba `Conferencia`) · `fuente` · `clase` · `parte` (`inicio`/`medio`/`fin` o vacío) · `n_oraciones` (oraciones completas; 0 en `parte`) · `n_palabras` · `origen` (de dónde sale la primera oración, para volver al texto) · `bloque` (`Indice` de la primera instancia que sale de la misma oración: los pedazos de una oración partida comparten `bloque`; en el resto es el propio `Indice`).

Regenerar: `python src/armar_instancias_final.py`.
"""
    (SALIDA / "README.md").write_text(texto, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, descartadas = armar_todo(args.seed)
    assert df["Indice"].is_unique and (df["Fragmento"].str.len() > 0).all()
    faltan = [(f, c) for f in FUENTES for c in CLASES if not ((df["fuente"] == f) & (df["clase"] == c)).any()]
    assert not faltan, f"clases sin instancias: {faltan}"

    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "instancias_final.csv").write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    escribir_readme(df, descartadas, args.seed)
    print(tabla_md(resumen_clases(df)))
    print(f"{len(df)} instancias, {int(df['n_palabras'].sum())} palabras -> {SALIDA}")


if __name__ == "__main__":
    main()
