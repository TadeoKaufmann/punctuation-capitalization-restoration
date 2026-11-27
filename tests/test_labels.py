"""Word labels extracted from punctuated text. Needs transformers (the BERT tokenizer is fetched on first import)."""
import pytest

pytest.importorskip("transformers")
try:
    import armar_instancias as ai
    import embeddings_palabras as ep
except OSError as error:   # tokenizer not cached and no network
    pytest.skip(f"BERT tokenizer unavailable: {error}", allow_module_level=True)


@pytest.mark.parametrize("word, expected", [
    ("casa", 0), ("Casa", 1), ("McDonald's", 2), ("iPhone", 2), ("ONU", 3), ("Y", 1), ("1985", 0),
])
def test_capitalization_classes(word, expected):
    assert ai.clase_capitalizacion(word) == expected


@pytest.mark.parametrize("suffix, expected", [("", ""), (",", ","), ("?)", "?"), ('".', "."), ("!", "")])
def test_closing_mark_is_first_supported_one(suffix, expected):
    assert ai.puntuacion_final(suffix) == expected


def test_words_and_labels_from_a_sentence():
    rows = ep.palabras_y_etiquetas("¿Venís mañana, Lucía? Sí, a la ONU.")
    assert [r["palabra"] for r in rows] == ["venís", "mañana", "lucía", "sí", "a", "la", "onu"]
    assert [r["punt_inicial"] for r in rows] == ["¿", "", "", "", "", "", ""]
    assert [r["punt_final"] for r in rows] == ["", ",", "?", ",", "", "", "."]
    assert [r["capitalizacion"] for r in rows] == [1, 0, 1, 1, 0, 0, 3]


def test_normalization_keeps_accents():
    assert ep.normalizar_texto("¿Qué pasó, acá?") == "qué pasó acá"


def test_demo_round_trip_rebuilds_the_original_text():
    from demo import clean, render
    text = "¿Venís mañana, Lucía? Sí, a la ONU."
    rows = ep.palabras_y_etiquetas(text)
    marks = {"": 0, ",": 1, ".": 2, "?": 3}
    rebuilt = render(clean(text), [int(r["punt_inicial"] == "¿") for r in rows],
                     [marks[r["punt_final"]] for r in rows], [r["capitalizacion"] for r in rows])
    assert rebuilt == text
