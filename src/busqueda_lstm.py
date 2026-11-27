"""
Búsqueda de hiperparámetros y entrenamiento final de la LSTM de puntuación y capitalización.

Es el mismo procedimiento de notebooks/lstm_puntuacion.ipynb, pero encadenado y reanudable:

    A      capas (L) y tamaño oculto (H)
    B      dropout, learning rate y weight decay (un eje a la vez, en ese orden)
    C      potencia de los pesos de clase (y, con --alphas, los pesos de cada cabeza)
    D      umbral por clase sobre las probabilidades (chequeado en mitades de validación no vistas)
    final  reentrena la mejor config con varias semillas y evalúa en TEST (una sola vez por semilla)

Cada etapa parte de la mejor config de la anterior. Un valor nuevo reemplaza al actual solo si mejora
el score de validación en más de --margen (por defecto 0,005): así no se persigue el ruido y, a igual
rendimiento, se queda el valor ya elegido.

Uso (desde cualquier carpeta; hace falta salidas/embeddings_palabras.npy y salidas/etiquetas_palabras.csv):

    python src/busqueda_lstm.py                        # todo: A B C D final
    python src/busqueda_lstm.py --etapas B C D         # solo esas (usan la config guardada de las anteriores)
    python src/busqueda_lstm.py --etapas final         # solo el reentrenamiento final
    python src/busqueda_lstm.py --rapido               # prueba de humo con datos y épocas mínimos

Reanudación y dataset nuevo: cada fila de salidas/busqueda_script.csv lleva una huella del dataset
(hash de las etiquetas + forma y muestra de los embeddings). Una corrida ya hecha solo se reutiliza si la
huella coincide, y la config elegida se guarda por huella en salidas/estado_busqueda.json. Si se
regeneran los embeddings, la huella cambia y todo se recalcula desde la config por defecto.
"""
import argparse
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

from protocolo import buscar_factores, dividir, f1_macro, guardar_predicciones, huella_split, reporte_split
from rutas import SALIDAS

warnings.filterwarnings("ignore", message="dropout option adds dropout")   # con L=1 el dropout de la LSTM no actúa

NOMBRES = ["punt_inicial", "punt_final", "capitalizacion"]
K_CLASES = {"punt_inicial": 2, "punt_final": 4, "capitalizacion": 4}
MAPA = {
    "punt_inicial":   {"": 0, "¿": 1},
    "punt_final":     {"": 0, ",": 1, ".": 2, "?": 3},
    "capitalizacion": {0: 0, 1: 1, 2: 2, 3: 3},
}

# Config por defecto: el punto de partida de la búsqueda
CFG_DEFECTO = dict(H=128, L=2, dropout=0.2, lr=1e-3, weight_decay=0.0, potencia=0.5,
                   a_ini=1.0, a_fin=1.0, a_cap=1.0, seed=0)

GRILLAS = {
    "A": {"L": [1, 2, 3], "H": [64, 128, 256]},
    "B": {"dropout": [0.1, 0.2, 0.3, 0.5], "lr": [3e-4, 1e-3, 3e-3], "weight_decay": [0.0, 1e-5, 1e-4]},
    "C": {"potencia": [0.0, 0.25, 0.5, 0.75, 1.0]},
    "alphas": [],                              # combinaciones completas (a_ini, a_fin, a_cap); vacío si se usa alphas_cabeza
    "alphas_cabeza": [0.5, 1.0, 2.0],          # peso de cada cabeza en la pérdida, una cabeza por vez (con --alphas)
}
GRILLAS_RAPIDO = {
    "A": {"L": [1, 2], "H": [128]},
    "B": {"dropout": [0.2, 0.3], "lr": [1e-3, 3e-3], "weight_decay": [0.0]},
    "C": {"potencia": [0.0, 0.5]},
    "alphas": [(1.0, 2.0, 2.0)],
}


GRILLAS_BI = {
    "A": {"L": [1, 2], "H": [64, 128, 256]},   # H por dirección
    "B": {"dropout": [0.2, 0.4], "lr": [1e-3, 3e-3], "weight_decay": [0.0, 1e-5]},
    "C": {"potencia": [0.25, 0.5, 0.75]},
    "alphas": [],
    "alphas_cabeza": [0.5, 1.0, 2.0],
}
GRILLAS_BI_RAPIDO = {
    "A": {"L": [1, 2], "H": [64]},
    "B": {"dropout": [0.2, 0.3], "lr": [1e-3, 3e-3], "weight_decay": [0.0]},
    "C": {"potencia": [0.25, 0.5]},
    "alphas": [],
}
CFG_BI = {**CFG_DEFECTO, "H": 64, "lr": 3e-3, "arq": "bilstm"}   # punto de partida de la búsqueda de la BiLSTM


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
class Datos:
    """Embeddings, etiquetas y los tres conjuntos (train / val / test), separados por documento (src/protocolo.py)."""

    def __init__(self, rapido=False):
        csv, npy = SALIDAS / "etiquetas_palabras.csv", SALIDAS / "embeddings_palabras.npy"
        self.X = np.load(npy)
        self.et = pd.read_csv(csv, keep_default_na=False)
        assert len(self.X) == len(self.et), (
            f"X tiene {len(self.X)} filas y las etiquetas {len(self.et)}: no son del mismo dataset")
        for col, mapa in MAPA.items():
            self.et[col + "_id"] = self.et[col].map(mapa)
        assert not self.et[[c + "_id" for c in NOMBRES]].isna().any().any(), "hay etiquetas fuera del mapa"
        self.huella = hashlib.md5(
            csv.read_bytes() + str(self.X.shape).encode() + self.X[::5000].tobytes()).hexdigest()[:8]

        # Mismo split para todos los modelos (src/protocolo.py): por documento (o bloque en paenza y sintetico),
        # estratificado. Reemplaza al split por instancia.
        self.idx_train, self.idx_val, self.idx_test = dividir(self.et)
        self.huella_split = huella_split(self.idx_train, self.idx_val, self.idx_test)
        (SALIDAS / "split_reporte.txt").write_text(
            reporte_split(self.et, self.idx_train, self.idx_val, self.idx_test), encoding="utf-8")

        self.seq_train, _ = self._secuencias(self.idx_train)
        self.seq_val, _ = self._secuencias(self.idx_val)
        self.seq_test, self.filas_test = self._secuencias(self.idx_test)   # filas_test: fila global de cada palabra
        if rapido:   # prueba de humo: pocos datos (los pesos de clase siguen saliendo de todo train)
            self.seq_train, self.seq_val = self.seq_train[:600], self.seq_val[:300]
            self.seq_test, self.filas_test = self.seq_test[:300], self.filas_test[:300]
        self.dl_val = hacer_dl(self.seq_val, mezclar=False)
        self.dl_test = hacer_dl(self.seq_test, mezclar=False)
        log(f"dataset {self.huella} | split {self.huella_split}: X {self.X.shape} | instancias train/val/test "
            f"{len(self.seq_train)}/{len(self.seq_val)}/{len(self.seq_test)}")

    def _secuencias(self, idx_filas):
        """(secuencias, filas): una tupla (x, y) por instancia (x (T,768) float, y (T,3) long) y, para cada una, las
        filas globales (posición en X y en etiquetas) de sus palabras."""
        sub = self.et.iloc[idx_filas].sort_values(["instancia_id", "posicion"])
        cols = [c + "_id" for c in NOMBRES]
        grupos = [g for _, g in sub.groupby("instancia_id", sort=False)]
        seqs = [(torch.from_numpy(self.X[g.index.to_numpy()]), torch.tensor(g[cols].values, dtype=torch.long)) for g in grupos]
        return seqs, [g.index.to_numpy() for g in grupos]

    def pesos_clase(self, nombre, potencia):
        """Un peso por clase calculado SOLO con train: (N / (K * n_c)) ** potencia."""
        K = K_CLASES[nombre]
        cuentas = np.bincount(self.et.iloc[self.idx_train][nombre + "_id"], minlength=K)
        return torch.tensor((cuentas.sum() / (K * cuentas)) ** potencia, dtype=torch.float32)


def collate(batch):
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True)                       # relleno 0.0
    y = pad_sequence(ys, batch_first=True, padding_value=-100)   # relleno -100 (ignore_index)
    return x, y, y[:, :, 0] != -100


class SamplerPorLargo:
    """Batches de instancias de largo parecido (menos relleno). Con mezclar=True se ordena por largo dentro de
    bloques de batch_size*bloque instancias elegidas al azar y se desordena el orden de los batches."""

    def __init__(self, largos, batch_size=32, bloque=50, mezclar=True, seed=0):
        self.largos = np.asarray(largos)
        self.batch_size, self.bloque, self.mezclar = batch_size, bloque, mezclar
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return math.ceil(len(self.largos) / self.batch_size)

    def __iter__(self):
        n, bs = len(self.largos), self.batch_size
        if not self.mezclar:
            orden = np.argsort(self.largos, kind="stable")
            for j in range(0, n, bs):
                yield orden[j:j + bs].tolist()
            return
        orden = self.rng.permutation(n)
        batches = []
        for i in range(0, n, bs * self.bloque):
            bl = orden[i:i + bs * self.bloque]
            bl = bl[np.argsort(self.largos[bl], kind="stable")]
            batches += [bl[j:j + bs] for j in range(0, len(bl), bs)]
        for k in self.rng.permutation(len(batches)):
            yield batches[k].tolist()


def hacer_dl(seqs, mezclar, seed=0):
    sampler = SamplerPorLargo([len(x) for x, _ in seqs], mezclar=mezclar, seed=seed)
    return DataLoader(seqs, batch_sampler=sampler, collate_fn=collate)


# ---------------------------------------------------------------------------
# Modelo, pérdida, entrenamiento y evaluación
# ---------------------------------------------------------------------------
class PuntLSTM(nn.Module):
    def __init__(self, D=768, H=128, L=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(D, H, num_layers=L, batch_first=True, dropout=dropout)
        self.head_ini = nn.Linear(H, 2)
        self.head_fin = nn.Linear(H, 4)
        self.head_cap = nn.Linear(H, 4)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head_ini(out), self.head_fin(out), self.head_cap(out)


class PuntBiLSTM(nn.Module):
    """LSTM bidireccional (modelo (c) del enunciado). H es el tamaño oculto POR dirección; las cabezas ven 2H."""

    def __init__(self, D=768, H=128, L=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(D, H, num_layers=L, batch_first=True, dropout=dropout, bidirectional=True)
        self.head_ini = nn.Linear(2 * H, 2)   # nada / ¿
        self.head_fin = nn.Linear(2 * H, 4)   # nada / , / . / ?
        self.head_cap = nn.Linear(2 * H, 4)   # 0 / 1 / 2 / 3

    def forward(self, x):
        T = x.shape[1]
        largos = (x != 0).any(dim=-1).sum(dim=1)   # palabras reales de cada instancia (el relleno es todo ceros)
        paquete = pack_padded_sequence(x, largos.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(paquete)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)   # (B, T, 2H); relleno = 0
        return self.head_ini(out), self.head_fin(out), self.head_cap(out)


# Se busca el nombre al construir (no al importar): así el notebook viejo que reemplazaba `PuntLSTM` sigue andando.
ARQUITECTURAS = {"lstm": lambda **k: PuntLSTM(**k), "bilstm": lambda **k: PuntBiLSTM(**k)}


def armar_perdidas(datos, cfg, device):
    ces = [nn.CrossEntropyLoss(weight=datos.pesos_clase(n, cfg["potencia"]).to(device)) for n in NOMBRES]
    return ces, (cfg["a_ini"], cfg["a_fin"], cfg["a_cap"])


def perdida(logits, y, ces, alphas):
    por_cabeza = [ces[i](logits[i].reshape(-1, logits[i].shape[-1]), y[:, :, i].reshape(-1)) for i in range(3)]
    return sum(a * l for a, l in zip(alphas, por_cabeza))


def entrenar_epoca(modelo, dl, opt, ces, alphas, device):
    modelo.train()
    suma = 0.0
    for xb, yb, _ in dl:
        total = perdida(modelo(xb.to(device)), yb.to(device), ces, alphas)
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        suma += total.item()
    return suma / len(dl)


@torch.no_grad()
def predecir_probs(modelo, dl, device):
    """Probabilidades (softmax) por cabeza, clases reales e índice de instancia de cada palabra real.
    Requiere un dl sin mezcla (se recorre el batch_sampler a la par del DataLoader)."""
    modelo.eval()
    P, Y, G = [[], [], []], [[], [], []], []
    for idxs, (xb, yb, mb) in zip(dl.batch_sampler, dl):
        logits = modelo(xb.to(device))
        mb_d = mb.to(device)
        for i in range(3):
            P[i].append(torch.softmax(logits[i], dim=-1)[mb_d].cpu().numpy())
            Y[i].append(yb[:, :, i][mb].numpy())
        G.append(torch.tensor(idxs).unsqueeze(1).expand_as(mb)[mb].numpy())
    return [np.concatenate(p) for p in P], [np.concatenate(y) for y in Y], np.concatenate(G)


def filas_de_predicciones(dl, filas_por_secuencia):
    """Fila global de cada palabra, en el MISMO orden en que predecir_probs las devuelve (batches del sampler,
    y dentro de cada batch las instancias en el orden del batch)."""
    return np.concatenate([filas_por_secuencia[i] for idxs in dl.batch_sampler for i in idxs])


def f1_por_cabeza(P, Y, factores=None):
    """F1 macro de cada cabeza; con `factores` (lista de 3 vectores) predice argmax(p * w)."""
    return [f1_macro(Y[i], (P[i] * (factores[i] if factores else 1.0)).argmax(1), K_CLASES[n])
            for i, n in enumerate(NOMBRES)]


# ---------------------------------------------------------------------------
# Una corrida con checkpoint y registro en CSV
# ---------------------------------------------------------------------------
CAMPOS_CFG = ["H", "L", "dropout", "lr", "weight_decay", "potencia", "a_ini", "a_fin", "a_cap", "seed"]


def entrenar(datos, cfg, n_epocas, device, verbose=False):
    """Entrena un modelo nuevo; evalúa en val cada época y devuelve (modelo con los pesos de la mejor época, info)."""
    torch.manual_seed(cfg["seed"])
    dl_train = hacer_dl(datos.seq_train, mezclar=True, seed=cfg["seed"])
    ces, alphas = armar_perdidas(datos, cfg, device)
    modelo = ARQUITECTURAS[cfg.get("arq", "lstm")](H=cfg["H"], L=cfg["L"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    mejor = {"score": -1.0, "epoca": 0}
    t0 = time.time()
    for ep in range(1, n_epocas + 1):
        perd_train = entrenar_epoca(modelo, dl_train, opt, ces, alphas, device)
        P, Y, _ = predecir_probs(modelo, datos.dl_val, device)
        f1 = f1_por_cabeza(P, Y)
        score = float(np.mean(f1))
        if score > mejor["score"]:
            mejor = {"score": score, "epoca": ep, "f1": f1, "perdida_train": perd_train,
                     "estado": {k: v.detach().clone() for k, v in modelo.state_dict().items()}}
        if verbose:
            log(f"  ep {ep:2d} | train {perd_train:.3f} | F1 val {f1[0]:.3f} / {f1[1]:.3f} / {f1[2]:.3f} | score {score:.3f}"
                + (" *" if mejor["epoca"] == ep else ""))
    modelo.load_state_dict(mejor.pop("estado"))
    mejor["segundos"] = round(time.time() - t0)
    return modelo, mejor


class Registro:
    """CSV de corridas (una fila por entrenamiento de búsqueda). Reutiliza las ya hechas con la misma huella."""

    def __init__(self, csv, huella):
        self.csv, self.huella = Path(csv), huella

    def buscar(self, cfg, n_epocas):
        if not self.csv.exists():
            return None
        d = pd.read_csv(self.csv)
        m = (d.dataset == self.huella) & (d.n_epocas == n_epocas)
        for c in CAMPOS_CFG:
            m &= np.isclose(d[c].astype(float), float(cfg[c]))
        return d[m].iloc[-1] if m.any() else None

    def agregar(self, etapa, cfg, n_epocas, info):
        fila = {"dataset": self.huella, "etapa": etapa, **{c: cfg[c] for c in CAMPOS_CFG}, "n_epocas": n_epocas,
                "mejor_epoca": info["epoca"], "score_val": round(info["score"], 4),
                **{"f1_" + n: round(info["f1"][i], 4) for i, n in enumerate(NOMBRES)}, "segundos": info["segundos"]}
        pd.DataFrame([fila]).to_csv(self.csv, mode="a", header=not self.csv.exists(), index=False)
        return pd.Series(fila)


def correr(ctx, etapa, cfg):
    """Devuelve la fila de resultados de `cfg`, entrenando solo si no está en el CSV (misma huella y épocas)."""
    fila = ctx.registro.buscar(cfg, ctx.n_epocas)
    if fila is None:
        _, info = entrenar(ctx.datos, cfg, ctx.n_epocas, ctx.device)
        fila = ctx.registro.agregar(etapa, cfg, ctx.n_epocas, info)
        nuevo = ""
    else:
        nuevo = " (ya hecha)"
    log(f"  {etapa}: " + ", ".join(f"{k}={cfg[k]}" for k in ("L", "H", "dropout", "lr", "weight_decay", "potencia",
                                                            "a_ini", "a_fin", "a_cap"))
        + f" -> score {fila.score_val:.3f} | F1 {fila.f1_punt_inicial:.3f} / {fila.f1_punt_final:.3f} / "
          f"{fila.f1_capitalizacion:.3f} | época {int(fila.mejor_epoca)}{nuevo}")
    return fila


def elegir(ctx, candidatos, actual):
    """candidatos: {valor: fila}. Devuelve el mejor valor, pero solo reemplaza a `actual` si lo supera en > margen."""
    mejor = max(candidatos, key=lambda v: candidatos[v].score_val)
    if candidatos[mejor].score_val - candidatos[actual].score_val <= ctx.margen:
        return actual
    return mejor


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------
def etapa_A(ctx, cfg, estado):
    g = ctx.grillas["A"]
    candidatos = {}
    for L in sorted(set(g["L"] + [cfg["L"]])):
        for H in sorted(set(g["H"] + [cfg["H"]])):
            candidatos[(L, H)] = correr(ctx, "A", {**cfg, "L": L, "H": H})
    cfg["L"], cfg["H"] = elegir(ctx, candidatos, (cfg["L"], cfg["H"]))
    return cfg


def etapa_B(ctx, cfg, estado):
    for eje, valores in ctx.grillas["B"].items():
        candidatos = {v: correr(ctx, "B", {**cfg, eje: v}) for v in sorted(set(valores + [cfg[eje]]))}
        cfg[eje] = elegir(ctx, candidatos, cfg[eje])
        log(f"  -> {eje} = {cfg[eje]}")
    return cfg


def etapa_C(ctx, cfg, estado):
    g = ctx.grillas["C"]
    candidatos = {v: correr(ctx, "C", {**cfg, "potencia": v}) for v in sorted(set(g["potencia"] + [cfg["potencia"]]))}
    cfg["potencia"] = elegir(ctx, candidatos, cfg["potencia"])
    log(f"  -> potencia = {cfg['potencia']}")
    if ctx.buscar_alphas:
        valores = ctx.grillas.get("alphas_cabeza")
        if valores:
            # Un eje a la vez, como la etapa B: cada cabeza prueba sus valores con las otras en su valor actual
            # (3 cabezas x 3 valores = 6 corridas nuevas, en vez de las 27 combinaciones).
            for clave in ("a_ini", "a_fin", "a_cap"):
                cand = {v: correr(ctx, "C", {**cfg, clave: v}) for v in sorted(set(valores + [cfg[clave]]))}
                cfg[clave] = elegir(ctx, cand, cfg[clave])
                log(f"  -> {clave} = {cfg[clave]}")
        else:
            actual = (cfg["a_ini"], cfg["a_fin"], cfg["a_cap"])
            cand = {a: correr(ctx, "C", {**cfg, "a_ini": a[0], "a_fin": a[1], "a_cap": a[2]})
                    for a in sorted(set(ctx.grillas["alphas"] + [actual]))}
            cfg["a_ini"], cfg["a_fin"], cfg["a_cap"] = elegir(ctx, cand, actual)
            log(f"  -> alphas = {(cfg['a_ini'], cfg['a_fin'], cfg['a_cap'])}")
    return cfg


def etapa_D(ctx, cfg, estado):
    """Reentrena la config (semilla 0) y mide cuánto sube el F1 con umbrales optimizados en una mitad de val y
    medidos en la otra. Decide si el umbral se usa (mejora media >= --min-ganancia-umbral)."""
    modelo, info = entrenar(ctx.datos, cfg, ctx.n_epocas, ctx.device)
    P, Y, G = predecir_probs(modelo, ctx.datos.dl_val, ctx.device)
    grupos = np.random.default_rng(0).permutation(np.unique(G))
    en_A = np.isin(G, grupos[: len(grupos) // 2])
    ganancias = []
    for i, n in enumerate(NOMBRES):
        K = K_CLASES[n]
        antes, despues = [], []
        for opt_m, med_m in [(en_A, ~en_A), (~en_A, en_A)]:
            w = buscar_factores(P[i][opt_m], Y[i][opt_m], K)
            antes.append(f1_macro(Y[i][med_m], P[i][med_m].argmax(1), K))
            despues.append(f1_macro(Y[i][med_m], (P[i][med_m] * w).argmax(1), K))
        ganancias.append(np.mean(despues) - np.mean(antes))
        log(f"  D: {n:15s} F1 sin umbral {np.mean(antes):.3f} -> con umbral {np.mean(despues):.3f}")
    estado["usar_umbral"] = bool(np.mean(ganancias) >= ctx.min_ganancia_umbral)
    estado["ganancia_umbral"] = round(float(np.mean(ganancias)), 4)
    log(f"  -> ganancia media {np.mean(ganancias):+.3f} | usar umbral: {estado['usar_umbral']}")
    return cfg


def etapa_final(ctx, cfg, estado):
    """Reentrena la config con varias semillas (checkpoint por val), ajusta los umbrales con val y mide en TEST."""
    usar = estado.get("usar_umbral", False)
    filas, mejor_global = [], None
    for seed in range(ctx.semillas_final):
        c = {**cfg, "seed": seed}
        log(f"  final: semilla {seed}, {ctx.n_epocas_final} épocas")
        modelo, info = entrenar(ctx.datos, c, ctx.n_epocas_final, ctx.device, verbose=True)
        Pv, Yv, _ = predecir_probs(modelo, ctx.datos.dl_val, ctx.device)
        factores = [buscar_factores(Pv[i], Yv[i], K_CLASES[n]) if usar else np.ones(K_CLASES[n])
                    for i, n in enumerate(NOMBRES)]
        Pt, Yt, _ = predecir_probs(modelo, ctx.datos.dl_test, ctx.device)   # único contacto con test
        f1_sin, f1_con = f1_por_cabeza(Pt, Yt), f1_por_cabeza(Pt, Yt, factores)
        fila = {"dataset": ctx.datos.huella, "seed": seed, "mejor_epoca": info["epoca"], "n_epocas": ctx.n_epocas_final,
                "score_val": round(info["score"], 4), "usa_umbral": usar,
                **{"test_" + n: round(f1_sin[i], 4) for i, n in enumerate(NOMBRES)},
                **{"test_umbral_" + n: round(f1_con[i], 4) for i, n in enumerate(NOMBRES)},
                "score_test": round(float(np.mean(f1_sin)), 4), "score_test_umbral": round(float(np.mean(f1_con)), 4),
                **{k: cfg[k] for k in CAMPOS_CFG if k != "seed"}}
        filas.append(fila)
        log(f"  final semilla {seed}: TEST F1 {f1_sin[0]:.3f} / {f1_sin[1]:.3f} / {f1_sin[2]:.3f} (score {np.mean(f1_sin):.3f})"
            + (f" | con umbral {f1_con[0]:.3f} / {f1_con[1]:.3f} / {f1_con[2]:.3f} (score {np.mean(f1_con):.3f})" if usar else ""))
        if mejor_global is None or info["score"] > mejor_global["score"]:   # se guarda el mejor por VAL, no por test
            mejor_global = {"score": info["score"], "estado": {k: v.cpu() for k, v in modelo.state_dict().items()},
                            "factores": [f.tolist() for f in factores], "seed": seed,
                            "preds": {n: (Pt[i] * factores[i]).argmax(1) for i, n in enumerate(NOMBRES)},
                            "filas": filas_de_predicciones(ctx.datos.dl_test, ctx.datos.filas_test)}
    df = pd.DataFrame(filas)
    df.to_csv(ctx.csv_final, mode="a", header=not ctx.csv_final.exists(), index=False)
    cols = ["test_" + n for n in NOMBRES] + ["score_test"] + (["test_umbral_" + n for n in NOMBRES] + ["score_test_umbral"] if usar else [])
    resumen = pd.DataFrame({"media": df[cols].mean(), "desvío": df[cols].std(ddof=0)}).round(3)
    log("  resumen en TEST (media y desvío entre semillas):")
    print(resumen.to_string())
    torch.save({"state_dict": mejor_global["estado"], "cfg": cfg, "factores": mejor_global["factores"],
                "usa_umbral": usar, "dataset": ctx.datos.huella, "seed": mejor_global["seed"]}, ctx.ruta_modelo)
    log(f"  modelo (mejor semilla por val: {mejor_global['seed']}) guardado en {ctx.ruta_modelo}")
    # predicciones de test de ese mismo modelo, para src/evaluacion.py (matrices de confusión, F1 por clase)
    ruta = guardar_predicciones(ctx.csv_final.parent, getattr(ctx, "nombre_modelo", "lstm"), mejor_global["filas"],
                                mejor_global["preds"], ctx.datos.huella,
                                {"seed": mejor_global["seed"], "usa_umbral": usar, "split": ctx.datos.huella_split})
    log(f"  predicciones de test guardadas en {ruta}")
    return cfg


ETAPAS = {"A": etapa_A, "B": etapa_B, "C": etapa_C, "D": etapa_D, "final": etapa_final}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=["lstm", "bilstm"], default="lstm", help="arquitectura (carpeta salidas/<modelo>/)")
    ap.add_argument("--etapas", nargs="+", default=["A", "B", "C", "D", "final"], choices=list(ETAPAS))
    ap.add_argument("--epocas", type=int, default=12, help="épocas por corrida de búsqueda (A, B, C, D)")
    ap.add_argument("--epocas-final", type=int, default=20, help="épocas del reentrenamiento final")
    ap.add_argument("--semillas-final", type=int, default=3)
    ap.add_argument("--margen", type=float, default=0.005, help="mejora mínima de score val para cambiar un valor")
    ap.add_argument("--min-ganancia-umbral", type=float, default=0.005, help="mejora media de F1 para usar el umbral")
    ap.add_argument("--alphas", action="store_true", help="en la etapa C, buscar también los pesos de cada cabeza")
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos datos, 2 épocas, grillas mínimas")
    ap.add_argument("--reiniciar", action="store_true", help="ignora la config guardada y parte de la por defecto")
    a = ap.parse_args()

    bi = a.modelo == "bilstm"
    carpeta = SALIDAS / (a.modelo + ("_rapido" if a.rapido else ""))
    carpeta.mkdir(exist_ok=True)
    ctx = argparse.Namespace(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        datos=Datos(rapido=a.rapido),
        grillas=(GRILLAS_BI_RAPIDO if a.rapido else GRILLAS_BI) if bi else (GRILLAS_RAPIDO if a.rapido else GRILLAS),
        n_epocas=2 if a.rapido else a.epocas, n_epocas_final=2 if a.rapido else a.epocas_final,
        semillas_final=2 if a.rapido else a.semillas_final, margen=a.margen,
        min_ganancia_umbral=a.min_ganancia_umbral, buscar_alphas=a.alphas,
        csv_final=carpeta / "resultados_finales.csv", ruta_modelo=carpeta / "modelo_final.pt", nombre_modelo=a.modelo)
    ctx.registro = Registro(carpeta / "busqueda_script.csv", ctx.datos.huella)
    log(f"device: {ctx.device} | etapas: {' '.join(a.etapas)}")

    ruta_estado = carpeta / "estado_busqueda.json"
    todos = json.loads(ruta_estado.read_text(encoding="utf8")) if ruta_estado.exists() else {}
    estado = {} if a.reiniciar else todos.get(ctx.datos.huella, {})
    cfg = {**(CFG_BI if bi else CFG_DEFECTO), **estado.get("cfg", {})}
    log("config de partida:", cfg)

    for nombre in a.etapas:
        log(f"=== etapa {nombre} ===")
        t0 = time.time()
        cfg = ETAPAS[nombre](ctx, cfg, estado)
        estado["cfg"] = cfg
        estado.setdefault("etapas_hechas", [])
        if nombre not in estado["etapas_hechas"]:
            estado["etapas_hechas"].append(nombre)
        todos[ctx.datos.huella] = estado                    # se guarda después de cada etapa: se puede cortar y retomar
        ruta_estado.write_text(json.dumps(todos, indent=1, ensure_ascii=False), encoding="utf8")
        log(f"etapa {nombre} lista en {(time.time() - t0) / 60:.1f} min | config: {cfg}")


if __name__ == "__main__":
    main()
