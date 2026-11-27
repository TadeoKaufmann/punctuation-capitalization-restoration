"""Restore punctuation and capitals in Spanish text with a trained LSTM or BiLSTM.

    python scripts/demo.py "che vos sabés a qué hora sale el tren mañana"
    python scripts/demo.py --file mi_texto.txt          # one instance per line
    python scripts/demo.py                              # a few built-in sentences

Needs salidas/<model>/modelo_final.pt, produced by `python src/busqueda_lstm.py --modelo bilstm --etapas final`
(see README). Weights are not included in the repository.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Written for this demo; none of them comes from the training corpus.
EXAMPLES = [
    "che vos sabés a qué hora sale el tren a mar del plata mañana",
    "no sé qué decirte la verdad me parece que mejor lo charlamos el lunes",
    "ayer fuimos con sofía al museo de bellas artes y después comimos una pizza en corrientes",
    "me podés pasar el informe antes de las cinco por favor",
    "la onu y la unión europea anunciaron un nuevo acuerdo en ginebra",
]


def clean(text: str) -> list:
    """Lowercase words without the marks the model predicts, the same normalization used in training."""
    from embeddings_palabras import palabras_y_etiquetas
    return [row["palabra"] for row in palabras_y_etiquetas(text.lower())]


def render(words, opening, closing, capital) -> str:
    """Rebuild the text. Class 2 (mixed case, e.g. iPhone) cannot be recovered from a class id alone,
    so it is shown with only the first letter in upper case."""
    out = []
    for word, ini, fin, cap in zip(words, opening, closing, capital):
        if cap == 3:
            word = word.upper()
        elif cap in (1, 2):
            word = word[:1].upper() + word[1:]
        out.append(("¿" if ini == 1 else "") + word + ["", ",", ".", "?"][fin])
    return " ".join(out)


class Restorer:
    def __init__(self, checkpoint: Path):
        import busqueda_lstm as bl
        from embeddings_palabras import cargar_bert
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        self.model = bl.ARQUITECTURAS[cfg.get("arq", "lstm")](H=cfg["H"], L=cfg["L"], dropout=cfg["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        sizes = (2, 4, 4)
        self.factors = ([np.asarray(f) for f in ck["factores"]] if ck.get("usa_umbral")
                        else [np.ones(k) for k in sizes])
        self.tokenizer, self.matrix = cargar_bert()

    def restore(self, text: str) -> str:
        from embeddings_palabras import embedding_palabra
        words = clean(text)
        if not words:
            return ""
        vectors = [embedding_palabra(w, self.tokenizer, self.matrix) for w in words]
        x = np.stack([v if v is not None else np.zeros(self.matrix.shape[1]) for v in vectors]).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x)[None])
        classes = [(torch.softmax(l[0], dim=-1).numpy() * f).argmax(1) for l, f in zip(logits, self.factors)]
        return render(words, *classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="a sentence or short passage in Spanish")
    ap.add_argument("--file", help="UTF-8 file with one instance per line")
    ap.add_argument("--model", default="bilstm", choices=["lstm", "bilstm"])
    ap.add_argument("--checkpoint", help="path to modelo_final.pt")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    from rutas import SALIDAS
    checkpoint = Path(a.checkpoint) if a.checkpoint else SALIDAS / a.model / "modelo_final.pt"
    if not checkpoint.exists():
        sys.exit(f"No model found at {checkpoint}. Train one first (see README, 'Reproducing').")

    if a.file:
        texts = [line for line in Path(a.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        texts = [a.text] if a.text else EXAMPLES
    restorer = Restorer(checkpoint)
    for text in texts:
        print("in :", " ".join(clean(text)))
        print("out:", restorer.restore(text), end="\n\n")


if __name__ == "__main__":
    main()
