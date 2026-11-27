"""
Arma datasets/fragmentos_todos.csv: los fragmentos de Paenza, conferencias y subtítulos en un solo CSV
(receta de RESUMEN.md, sección 9).

Columnas de salida:
  Indice       único en todo el archivo (Paenza 1..3343, conferencias 100001.., subtítulos 200001..)
  Fragmento    texto, tal como viene de cada fuente (sin filtrar)
  Conferencia  grupo para partir train/test: la conferencia, la película, o "paenza" (no hay dato de libro)
  fuente       "paenza" | "conferencias" | "subtitulos"

Uso (desde cualquier carpeta):
  python src/consolidar_fragmentos.py                       # regenera datasets/fragmentos_todos.csv
  python src/consolidar_fragmentos.py --salida salidas/fragmentos_todos.csv
  python src/consolidar_fragmentos.py --verificar           # compara con el CSV actual, no escribe nada

Protecciones:
  - Si la salida ya existe y es igual a lo que se armaría, no se toca.
  - Si existe y es distinta, NO se pisa salvo que se pase --sobrescribir (se muestra qué cambiaría).
  - Se escribe siempre con fin de línea "\\n" (como el blob versionado en git), sin importar el sistema;
    en Windows con core.autocrlf=true el archivo de trabajo se ve con CRLF y la comparación lo tiene en cuenta.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from rutas import DATASETS

COLUMNAS = ["Indice", "Fragmento", "Conferencia", "fuente"]
GRUPO_PAENZA = "paenza"   # Paenza no trae libro ni capítulo: un único grupo


def leer_fuente(ruta: Path, fuente: str, con_conferencia: bool) -> pd.DataFrame:
    if not ruta.exists():
        sys.exit(f"Falta {ruta} (fuente '{fuente}').")
    df = pd.read_csv(ruta)
    requeridas = ["Indice", "Fragmento"] + (["Conferencia"] if con_conferencia else [])
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        sys.exit(f"{ruta.name}: faltan las columnas {faltan} (tiene {list(df.columns)}).")
    if df["Indice"].isna().any():
        sys.exit(f"{ruta.name}: hay filas sin Indice.")
    if not con_conferencia:
        df["Conferencia"] = GRUPO_PAENZA
    df["fuente"] = fuente
    return df


def consolidar(paenza: Path, conferencias: Path, subtitulos: Path) -> pd.DataFrame:
    partes = [
        leer_fuente(paenza, "paenza", con_conferencia=False),
        leer_fuente(conferencias, "conferencias", con_conferencia=True),
        leer_fuente(subtitulos, "subtitulos", con_conferencia=True),
    ]
    todos = pd.concat(partes)[COLUMNAS]
    repetidos = todos["Indice"][todos["Indice"].duplicated(keep=False)]
    if len(repetidos):
        rangos = {p["fuente"].iloc[0]: (int(p["Indice"].min()), int(p["Indice"].max())) for p in partes}
        sys.exit(f"Hay {repetidos.nunique()} Indice repetidos entre las fuentes (rangos: {rangos}); "
                 "los Indice tienen que ser únicos porque son el instancia_id.")
    vacios = int(todos["Fragmento"].isna().sum())
    if vacios:
        print(f"Aviso: {vacios} filas con Fragmento vacío (se conservan, como en la receta original).")
    return todos


def a_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def resumen(df: pd.DataFrame, titulo: str) -> None:
    print(titulo)
    g = df.groupby("fuente").agg(filas=("Indice", "size"), indice_min=("Indice", "min"),
                                 indice_max=("Indice", "max"), grupos=("Conferencia", "nunique"))
    print(g.reindex(["paenza", "conferencias", "subtitulos"]).dropna(how="all").to_string())
    print(f"  total: {len(df)} filas")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paenza", default=str(DATASETS / "fragmentos.csv"))
    ap.add_argument("--conferencias", default=str(DATASETS / "fragmentos_conferencias.csv"))
    ap.add_argument("--subtitulos", default=str(DATASETS / "fragmentos_subtitulos.csv"))
    ap.add_argument("--salida", default=str(DATASETS / "fragmentos_todos.csv"))
    ap.add_argument("--sobrescribir", action="store_true", help="permite pisar una salida existente que sea distinta")
    ap.add_argument("--verificar", action="store_true",
                    help="no escribe: dice si la salida actual es idéntica a lo que se armaría (código de salida 0 = idéntica)")
    args = ap.parse_args()

    salida = Path(args.salida)
    nuevo = consolidar(Path(args.paenza), Path(args.conferencias), Path(args.subtitulos))
    contenido = a_bytes(nuevo)
    resumen(nuevo, "Consolidado:")

    existe = salida.exists()
    # El archivo de trabajo puede tener CRLF (autocrlf en Windows); el contenido versionado usa LF
    actual = salida.read_bytes().replace(b"\r\n", b"\n") if existe else None
    igual = existe and actual == contenido

    if args.verificar:
        print(f"{salida}: {'IDÉNTICO' if igual else ('DIFIERE' if existe else 'NO EXISTE')} "
              f"(armado: {len(contenido)} bytes{'' if not existe else f', actual: {len(actual)} bytes'})")
        return 0 if igual else 1

    if igual:
        print(f"{salida} ya está al día; no se escribe.")
        return 0
    if existe and not args.sobrescribir:
        resumen(pd.read_csv(salida), f"{salida} existe y sería distinto. Hoy tiene:")
        print("No se pisa. Para hacerlo, repetir con --sobrescribir (o elegir otra --salida).")
        return 2

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(contenido)   # binario: fuerza "\n" aunque estemos en Windows
    print(f"Escrito {salida} ({len(contenido)} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
