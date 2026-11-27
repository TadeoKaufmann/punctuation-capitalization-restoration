"""The shared split and metrics, on synthetic data."""
import numpy as np
import pandas as pd
import pytest

import protocolo as pr


def fake_labels(n_docs=60, words_per_doc=80, seed=0):
    rng = np.random.default_rng(seed)
    n = n_docs * words_per_doc
    doc = np.repeat(np.arange(n_docs), words_per_doc)
    source = np.where(doc < 40, "subtitulos", "paenza")
    return pd.DataFrame({
        "documento": np.where(doc < 40, doc.astype(str), "paenza"),
        "bloque": doc * 10 + rng.integers(0, 3, n),
        "fuente": source,
        "punt_inicial": rng.choice(["", "¿"], n, p=[0.95, 0.05]),
        "punt_final": rng.choice(["", ",", ".", "?"], n, p=[0.8, 0.1, 0.07, 0.03]),
        "capitalizacion": rng.choice([0, 1, 2, 3], n, p=[0.85, 0.12, 0.015, 0.015]),
    })


def test_split_is_a_partition_without_shared_groups():
    et = fake_labels()
    train, val, test = pr.dividir(et)
    assert sorted(np.concatenate([train, val, test])) == list(range(len(et)))
    groups = pr.asignar_grupo(et)
    sets = [set(groups[i]) for i in (train, val, test)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_split_is_deterministic():
    et = fake_labels()
    assert pr.huella_split(*pr.dividir(et)) == pr.huella_split(*pr.dividir(et))


def test_single_document_sources_are_grouped_by_sentence():
    et = fake_labels()
    groups = pr.asignar_grupo(et)
    assert all(g.startswith("bloque:") for g in groups[et["fuente"] == "paenza"])
    assert all(g.startswith("doc:") for g in groups[et["fuente"] == "subtitulos"])


def test_missing_group_columns_are_rejected():
    with pytest.raises(AssertionError):
        pr.asignar_grupo(fake_labels().drop(columns=["documento"]))


def test_class_factors_never_lower_validation_macro_f1():
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1], 2000, p=[0.9, 0.1])
    p1 = np.clip(0.15 + 0.2 * y + rng.normal(0, 0.08, 2000), 0, 1)
    probs = np.column_stack([1 - p1, p1])
    weights = pr.buscar_factores(probs, y, 2)
    assert pr.f1_macro(y, (probs * weights).argmax(1), 2) > pr.f1_macro(y, probs.argmax(1), 2)


def test_confusion_matrix_rows_are_recalls():
    m = pr.matriz_confusion([0, 0, 1, 1], [0, 1, 1, 1], 2)
    np.testing.assert_allclose(m, [[0.5, 0.5], [0.0, 1.0]])
