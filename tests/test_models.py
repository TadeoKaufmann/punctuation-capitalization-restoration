"""Network shapes and padding behaviour, handcrafted features, token alignment at inference and the bootstrap."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import atributos_rf
import inferir
from bootstrap_ci import bootstrap


@pytest.mark.parametrize("arch", ["lstm", "bilstm"])
def test_padding_does_not_change_predictions(arch):
    torch = pytest.importorskip("torch")
    from busqueda_lstm import ARQUITECTURAS
    torch.manual_seed(0)
    model = ARQUITECTURAS[arch](H=16, L=2, dropout=0.0).eval()
    x = torch.randn(1, 5, 768)
    padded = torch.cat([x, torch.zeros(1, 4, 768)], dim=1)
    with torch.no_grad():
        alone, with_padding = model(x), model(padded)
    assert [t.shape[-1] for t in alone] == [2, 4, 4]
    for a, b in zip(alone, with_padding):
        torch.testing.assert_close(a, b[:, :5], atol=1e-5, rtol=1e-5)


def test_parameter_counts():
    pytest.importorskip("torch")
    from busqueda_lstm import PuntBiLSTM, PuntLSTM
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(PuntLSTM(H=128, L=2)) == 593_162
    assert count(PuntBiLSTM(H=64, L=2)) == 527_626


def test_feature_window_does_not_cross_instances():
    df = pd.DataFrame({"instancia_id": [1, 1, 2, 2], "_norm": ["qué", "hora", "es", "pero"]})
    X = atributos_rf.construir_atributos(df, vocab=["qué", "es"]).toarray()
    width = len(atributos_rf.atributos_palabra("a", 0, 1))
    previous = X[:, width:2 * width]                 # offset -1 is the second of the five dense blocks
    assert (previous[0] == -1).all() and (previous[2] == -1).all()
    assert not (previous[1] == -1).all() and not (previous[3] == -1).all()


def test_subword_tokens_are_grouped_into_words_and_labels_placed_on_the_edges():
    tokens = pd.DataFrame({"instancia_id": [1, 1, 1, 2], "token_id": [1, 2, 3, 4],
                           "token": ["cu", "##ando", "ven", "##is"]})
    words = inferir.agrupar_palabras(tokens)
    assert list(words) == [0, 0, 1, 2]               # a word never crosses instances
    classes = {"punt_inicial": np.array([1, 0, 0]), "punt_final": np.array([3, 1, 2]),
               "capitalizacion": np.array([1, 0, 0])}
    out = inferir.armar_salida(tokens, words, classes)
    assert list(out["punt_inicial"]) == ["¿", "", "", ""]
    assert list(out["punt_final"]) == ["", "?", ",", "."]
    assert list(out["capitalización"]) == [1, 1, 0, 0]


def test_bootstrap_point_estimate_matches_sklearn():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 50)
    labels = {"punt_final": rng.integers(0, 4, len(groups))}
    preds = {"m": {"punt_final": np.where(rng.random(len(groups)) < 0.7, labels["punt_final"], 0)}}
    samples, point = bootstrap(groups, labels, preds, n_boot=200)
    expected = f1_score(labels["punt_final"], preds["m"]["punt_final"], average="macro")
    assert point["m"] == pytest.approx(expected)
    assert samples["m"].shape == (200,)
    assert np.percentile(samples["m"], 2.5) < expected < np.percentile(samples["m"], 97.5)
