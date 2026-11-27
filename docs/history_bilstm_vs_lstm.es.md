# LSTM vs LSTM bidireccional sobre `instancias_final` (2026-11-17)

Búsqueda de hiperparámetros y evaluación final de las dos redes con el **mismo dataset, split y protocolo**. Corrido en Colab (GPU T4) con `src/busqueda_lstm.py` y `notebooks/bilstm_puntuacion.ipynb`. Los CSV de cada corrida están en `salidas/resultados_colab/` (local, fuera de git).

## Dataset y protocolo

- `datasets/instancias_final/instancias_final.csv` (huella del dataset `bc1c0570`): 42.363 instancias, **534.841 palabras** (Paenza, conferencias, subtítulos y sintético). Embeddings: promedio de los tokens BERT de cada palabra, 768 dimensiones.
- Split **por `instancia_id`**: test 20 % (`random_state=42`) y, del resto, validación 15 % (`random_state=0`). Es el mismo para las dos redes.
- Métrica: **F1 macro por tarea a nivel palabra** (inicial 2 clases, final 4, capitalización 4). El score de selección es el promedio de las tres.
- Se elige por score de **validación**; un valor nuevo reemplaza al actual solo si lo supera por más de 0,005. El test se mide una sola vez, con 3 semillas, al final.
- Búsqueda por etapas: A (capas `L` y tamaño `H`), B (dropout, lr, weight decay), C (`potencia` de los pesos de clase y pesos de cada cabeza, `alphas`), D (umbral por clase), final.

## Configuración elegida

| | LSTM | BiLSTM |
|---|---|---|
| `H` (por dirección en la BiLSTM) | 128 | **256** |
| `L` (capas) | 2 | 2 |
| dropout | 0,5 | 0,2 |
| lr | 1e-3 | 3e-3 |
| weight decay | 0 | 0 |
| `potencia` | 0,5 | 0,5 |
| alphas (inicial, final, cap.) | 1, 1, 1 | 1, 1, **2** |
| umbral por clase (etapa D) | **sí** (ganancia +0,0144) | no (ganancia −0,0018) |
| parámetros | 593.162 | ≈ 3,68 M |

## Resultado en test (media y desvío entre 3 semillas, 20 épocas)

| F1 macro | LSTM | LSTM + umbral | BiLSTM | BiLSTM − LSTM | BiLSTM − LSTM + umbral |
|---|---|---|---|---|---|
| puntuación inicial | 0,711 ± 0,010 | 0,738 ± 0,004 | **0,809 ± 0,001** | +0,098 | +0,071 |
| puntuación final | 0,497 ± 0,010 | 0,504 ± 0,005 | **0,680 ± 0,005** | +0,183 | +0,176 |
| capitalización | 0,702 ± 0,024 | 0,715 ± 0,007 | **0,738 ± 0,004** | +0,036 | +0,023 |
| **score (promedio)** | 0,637 ± 0,008 | 0,653 ± 0,003 | **0,742 ± 0,003** | **+0,105** | **+0,089** |

**La BiLSTM gana en las tres tareas**, por mucho más que el desvío entre semillas (0,001 a 0,024). La LSTM con umbral es su versión más favorecida, y aun así queda 0,09 abajo.

- **Puntuación final** es la tarea más difícil y donde más se nota la bidireccionalidad (+0,18): para decidir si hay coma, punto o `?` después de una palabra ayuda saber qué viene después (`que`, `pero`, `y`, el inicio de otra pregunta).
- **Capitalización** es la que menos cambia (+0,04): depende sobre todo de la palabra en sí y de si abre una oración. Las clases 2 y 3 tienen muy pocas palabras, y por eso es la tarea más ruidosa entre semillas.

## Qué dice la búsqueda (validación, 1 semilla por config)

**BiLSTM** (17 corridas): 
- **`L = 2` rinde mejor que `L = 1`** (score 0,73 a 0,74 contra 0,71 a 0,73): es la señal más clara de la etapa A.
- `H`: 64 → 0,7317, 128 → 0,7326, 256 → 0,7443. La mejora de 256 es de 0,012, cerca del ruido.
- Etapa B: ni dropout 0,4, ni lr 1e-3, ni weight decay 1e-5 mejoraron a la base (0,2 / 3e-3 / 0). El weight decay empeoró (0,7228).
- Etapa C: `potencia` 0,25 (0,7474) y 0,75 (0,7286) no superaron el margen; se quedó 0,5. De las alphas, solo **`a_cap = 2` mejoró claramente** (0,7551, +0,011 sobre la base). Las demás quedaron dentro del ruido.
- Etapa D: el umbral no ayudó (ganancia −0,0018), así que no se usa.

**LSTM** (26 corridas): `L`, `H` y las alphas casi no importan (meseta alrededor de 0,63). Lo que más mueve el score es **dropout 0,5** (0,6492 contra 0,6337 con 0,2) y la `potencia` (0 y 1 empeoran claro: 0,588 y 0,563; la mejor es 0,5). El umbral por clase sí sube su F1 (+0,016).

Diferencias de score menores a ~0,015 no son concluyentes: las clases 2 y 3 de capitalización tienen pocas palabras en validación y cada config se corrió con una sola semilla.

## ¿El resultado es por la dirección o por el tamaño?

La BiLSTM elegida es unas 6 veces más grande que la LSTM (3,68 M contra 0,59 M parámetros), así que parte de la diferencia podría ser tamaño. Una comparación a tamaño parecido sale de la etapa A: **BiLSTM `H = 64`, `L = 2`** (527.626 parámetros, casi igual a la LSTM) saca **0,7317 en validación**, contra 0,634 de la LSTM (`H = 128`, `L = 2`). La ventaja se mantiene con el mismo tamaño, así que se debe en lo esencial a ver las palabras de los dos lados. Esa comparación usa una semilla y los lr de partida de cada búsqueda (3e-3 y 1e-3); la LSTM con 3e-3 y `H = 128` dio 0,6362, o sea que no cambia la conclusión.

## Advertencias

- **El test es optimista.** El split es por `instancia_id`: pedazos de una misma oración (clase `parte`) y fragmentos de la misma película pueden caer en train y en test. Los números absolutos probablemente estén inflados; la comparación entre las dos redes bajo el mismo split sigue siendo válida. (Decisión del equipo del 2026-11-16, ver `DECISIONS.md`.)
- Después de elegir el mejor epoch por validación se entrenó 20 épocas: la pérdida de train sigue bajando (0,13 en la BiLSTM) mientras la de validación se estanca desde la época ~10, o sea que hay sobreajuste y el checkpoint por validación importa.
- Los números no se pueden comparar con los del Random Forest ni con corridas anteriores: son de otro corpus.
- Las primeras 3 corridas de la BiLSTM (`L = 1`) tardaron 8 a 31 minutos porque se corrieron en CPU antes de conectar la GPU; el resto, unos 2 minutos cada una. Los resultados no dependen del dispositivo.

## Tiempos

BiLSTM: 17 corridas de búsqueda + etapa final de 9,9 min. LSTM: 26 corridas (~75 s cada una) + etapa final de 6,7 min. Con GPU, una corrida de 12 épocas de la BiLSTM `H = 256` tarda unos 2 minutos.

## Para completar

- Falta el script de inferencia (CSV por token) y el Random Forest (con y sin embeddings) sobre `instancias_final`.
- Pendiente de acordar con Tadeo y Mauricio: si se parte train/test por película o fuente, y si se repite todo con ese split.
