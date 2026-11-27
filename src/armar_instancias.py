"""
Arma instancias etiquetadas (formato de "Salida esperada" del TP) a partir de
textos con puntuación y mayúsculas.

Idea: se recorre el texto ORIGINAL palabra por palabra, se sacan las etiquetas
de cada palabra (¿ inicial, , . ? final, clase de capitalización) y recién
después se normaliza la palabra (minúsculas, sin puntuación) y se tokeniza con
BERT. Así la etiqueta de la palabra se copia a sus sub-tokens.
"""
import re
from pathlib import Path
import pandas as pd
from transformers import BertTokenizer

from rutas import DATASETS, SALIDAS

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

PUNT_FINALES = {",", ".", "?"}
# Separa cada "palabra" en: puntuación de adelante, núcleo, puntuación de atrás
PALABRA_RE = re.compile(r"^(\W*)(.*?)(\W*)$")


def clase_capitalizacion(palabra: str) -> int:
    letras = [c for c in palabra if c.isalpha()]
    if not letras or all(c.islower() for c in letras):
        return 0
    if all(c.isupper() for c in letras):
        return 3 if len(letras) > 1 else 1
    if letras[0].isupper() and all(c.islower() for c in letras[1:]):
        return 1
    return 2


def puntuacion_final(sufijo: str) -> str:
    # Nos quedamos con el primer signo relevante de la cola (ej. "?)" -> "?")
    for c in sufijo:
        if c in PUNT_FINALES:
            return c
    return ""


def etiquetar_texto(texto: str, instancia_id: int) -> list[dict]:
    filas = []
    for palabra in texto.split():
        prefijo, nucleo, sufijo = PALABRA_RE.match(palabra).groups()
        if not nucleo:          # "palabra" que es solo puntuación (ej. " - ")
            continue
        p_ini = "¿" if "¿" in prefijo else ""
        p_fin = puntuacion_final(sufijo)
        cap = clase_capitalizacion(nucleo)

        # Lo que va a ver el modelo: minúsculas y sin puntuación pegada
        tokens = tokenizer.tokenize(nucleo.lower())
        ids = tokenizer.convert_tokens_to_ids(tokens)

        for i, (tok, tid) in enumerate(zip(tokens, ids)):
            filas.append({
                "instancia_id": instancia_id,
                "token_id": tid,
                "token": tok,
                "punt_inicial": p_ini if i == 0 else "",
                "punt_final": p_fin if i == len(tokens) - 1 else "",
                "capitalización": cap,
            })
    return filas


# ---------------------------------------------------------------------------
# Lectura y filtrado de fragmentos (fragmentos.csv: columnas Indice, Fragmento)
# ---------------------------------------------------------------------------
INDICE_DESDE = 100        # los primeros fragmentos son carátula/prólogo
MAX_DIGITOS_PALABRA = 4   # una palabra con >= 4 dígitos (ej. "1976", "40.000.000") descarta el fragmento
MAX_FRAC_NUMERICAS = 0.2  # si más de este % de las palabras tiene algún dígito, es una cuenta/tabla

# Operación aritmética: número/letra, operador, número/letra (ej. "3 x 4", "1/2", "a + b", "x = 5").
# Se exige algún dígito o una letra suelta para no matar guiones de diálogo (— ) ni "a-b" de palabras.
RE_CUENTA = re.compile(
    r"(?:\d\s*[+\-−–x×*/÷=^]\s*\d)"            # 3+4, 10 x 10, 1/2, 2^3
    r"|(?:\b[a-zA-Z]\s*[+\-−×*/÷=]\s*[a-zA-Z0-9]\b)"  # a + b, x = 5 (variables de una letra)
    r"|(?:\s[+*/×÷=]\s)"                          # operador suelto entre espacios
)

# Encabezados/pies de página que el PDF dejó pegados en medio de una oración, ej.
# "© Siglo Veintiuno Editores 218 ADRIÁN PAENZA", "216 ADRIÁN PAENZA MATEMÁTICA… ¿ESTÁS AHÍ? EPISODIO 3,14 217"
# o "MATEMÁTICA… ¿ESTÁS AHÍ? EPISODIO 2 27". El episodio puede ser "2" o "3,14" y a veces el
# fragmento empieza justo en "EPISODIO 2 23" (cortado antes de "MATEMÁTICA…"): esa forma sola
# se borra con la última alternativa. Distingue mayúsculas, así que no toca "episodio 3" de la prosa.
# Se BORRAN (la oración sigue siendo válida) en vez de descartar el fragmento entero.
RE_ENCABEZADO = re.compile(
    r"©\s*(?:Siglo Veintiuno Editores|\d{4}[^.]*?)?\s*\d*\s*(?:ADRIÁN PAENZA)?"
    r"|(?:\d+\s+ADRIÁN PAENZA\s+)?MATEMÁTICA\W+¿?ESTÁS AHÍ\??\s*(?:EPISODIO\s*\d+(?:,\d+)?)?\s*\d*"
    r"|\bEPISODIO\s+\d+(?:,\d+)?\s+\d+\b",
)

# Texto editorial que no es prosa del libro: ISBN, copyright, URLs, índices con puntos "....",
# encabezados de página ("© Siglo Veintiuno Editores 10 ADRIÁN PAENZA"), contratapa, etc.
RE_EDITORIAL = re.compile(
    r"ISBN|©|www\.|https?://|\.{4,}|(?:\. ){4,}|Siglo Veintiuno|Siglo XXI|"
    r"derechos reservados|Impreso en|IMPRESO|Colecci[oó]n|Editores|Editorial|"
    r"CDD|ADRI[AÁ]N PAENZA|MATEM[AÁ]TICA\W+¿?ESTÁS AHÍ",
    flags=re.IGNORECASE,
)


def filtrar_fragmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Saca de `df` (columna Fragmento) los textos editoriales y los de cuentas."""
    df = df.copy()
    df["Fragmento"] = df["Fragmento"].str.replace(RE_ENCABEZADO, " ", regex=True).str.split().str.join(" ")
    frag = df["Fragmento"]

    # Dígitos por palabra -> detecta números largos y fragmentos llenos de números
    palabras = frag.str.split()
    digitos_max = palabras.apply(lambda ps: max((sum(c.isdigit() for c in w) for w in ps), default=0))
    frac_numericas = palabras.apply(
        lambda ps: sum(any(c.isdigit() for c in w) for w in ps) / max(len(ps), 1)
    )

    es_numero_largo = digitos_max >= MAX_DIGITOS_PALABRA
    es_tabla_numeros = frac_numericas > MAX_FRAC_NUMERICAS
    es_cuenta = frag.str.contains(RE_CUENTA)
    es_editorial = frag.str.contains(RE_EDITORIAL)

    descartar = es_numero_largo | es_tabla_numeros | es_cuenta | es_editorial
    print(f"Fragmentos: {len(df)} | números largos: {es_numero_largo.sum()} | "
          f"muchos números: {es_tabla_numeros.sum()} | cuentas: {es_cuenta.sum()} | "
          f"editorial: {es_editorial.sum()} | quedan: {(~descartar).sum()}")
    return df[~descartar]  # ya con los encabezados borrados


def armar_dataset(ruta_csv: Path = DATASETS / "fragmentos.csv", desde: int = INDICE_DESDE, filtrar: bool = True) -> pd.DataFrame:
    """Lee fragmentos.csv, se queda con Indice >= `desde`, filtra (si `filtrar`) y etiqueta.

    `filtrar=False` salta los filtros de Paenza (cuentas, editorial); los fragmentos de
    scrapear_conferencias.py ya vienen filtrados.

    `instancia_id` es el Indice original del fragmento, así se puede volver al texto.
    """
    df = pd.read_csv(ruta_csv)
    df = df[df["Indice"] >= desde].dropna(subset=["Fragmento"])
    if filtrar:
        df = filtrar_fragmentos(df)

    filas = []
    for indice, texto in zip(df["Indice"], df["Fragmento"]):
        filas.extend(etiquetar_texto(texto, int(indice)))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = armar_dataset()
    print(df.head(30).to_string())
    # Archivo aparte para no pisar el instancias.csv de ejemplo; pandas pone "," entre comillas
    df.to_csv(SALIDAS / "instancias_fragmentos.csv", index=False)

    # Conferencias de la Casa Rosada (scrapear_conferencias.py), sin los filtros de Paenza
    if (DATASETS / "fragmentos_conferencias.csv").exists():
        dfc = armar_dataset(DATASETS / "fragmentos_conferencias.csv", desde=0, filtrar=False)
        dfc.to_csv(SALIDAS / "instancias_conferencias.csv", index=False)
        print(f"Conferencias: {dfc['instancia_id'].nunique()} instancias, {len(dfc)} tokens")
