# datasets/

Only code lives here. The corpora are not distributed; see [docs/data.md](../docs/data.md) for what to put in this
folder.

`subtitulos/scripts/`:

| Script | What it does |
|---|---|
| `resolver_peliculas.py` | resolves film titles to OpenSubtitles entries, trying Spanish and English titles |
| `listar_candidatas.py` | lists which candidate films have Spanish subtitles |
| `descargar_subtitulos.py` | picks one file per film with a quality score and downloads it, respecting the daily quota |
| `analizar_subtitulos.py` | reports language, `¿`/`?` ratio and voseo frequency of `.srt` files |

Credentials go in a `.env` file (copy `.env.example`).
