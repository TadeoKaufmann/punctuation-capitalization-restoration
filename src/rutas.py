"""Rutas del repo: los scripts funcionan sin importar desde qué carpeta se corran."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATASETS = RAIZ / "datasets"            # fragmentos*.csv e instancias.csv (versionados)
SUBTITULOS = DATASETS / "subtitulos"    # scripts de OpenSubtitles, listas y subs/ (.srt)
SALIDAS = RAIZ / "salidas"              # generados que se regeneran (fuera de git)
SALIDAS.mkdir(exist_ok=True)
