# Random Forest — baseline, optimización y resultados finales

**Trabajo Práctico II — Aprendizaje Automático II (UBA)**  
**Notebook asociado:** [`random_forest_pipeline_completo.ipynb`](../notebooks/exploration/random_forest_pipeline_completo.ipynb)

## 1. Objetivo

Reconstruir tres atributos por palabra de un texto normalizado:

1. **Puntuación inicial:** ausencia de signo o `¿`.
2. **Puntuación final:** ausencia de signo, `,`, `.` o `?`.
3. **Capitalización:** clases `0`, `1`, `2` y `3`.

Se entrenan **tres clasificadores Random Forest independientes**, uno por tarea. El objetivo principal de selección es **macro F1 a nivel palabra**: cada clase aporta el mismo peso al promedio, independientemente de su frecuencia. La accuracy se informa como métrica complementaria.

El pipeline utiliza **embeddings estáticos de BERT multilingüe (`bert-base-multilingual-cased`)**, de 768 dimensiones por palabra, obtenidos promediando embeddings de entrada de sus subtokens. No son embeddings contextualizados: el bosque recibe cada palabra individualmente, sin las palabras vecinas.

## 2. Datos y metodología

- **Total:** 500.973 palabras.
- **Train original:** 400.067 palabras.
- **Test reservado:** 100.906 palabras.
- **Subtrain para búsqueda:** 318.900 palabras.
- **Validación para búsqueda:** 81.167 palabras.
- **Separación:** por `instancia_id`/fragmento, para evitar que palabras del mismo fragmento aparezcan a la vez en entrenamiento y evaluación.
- **Selección:** hiperparámetros y reglas de decisión elegidos con validación, **sin consultar el test**.
- **Evaluación final:** reentrenamiento de los tres bosques sobre las 400.067 palabras del train original y evaluación en el test reservado con las reglas previamente seleccionadas.

## 3. Baseline del repositorio

Se reprodujo la configuración inicial, con tres bosques separados:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

| Tarea | Macro F1 baseline (test) | Accuracy baseline (test) |
|---|---:|---:|
| Puntuación inicial | 0,5249 | 84,22 % |
| Puntuación final | 0,3577 | 63,59 % |
| Capitalización | 0,4283 | 76,14 % |

**Macro F1 promedio:** 0,4370.  
**Accuracy conjunto:** 44,22 % de las palabras con las tres etiquetas correctas simultáneamente.

## 4. Búsqueda de hiperparámetros

Se realizó una exploración inicial de **12 configuraciones**, con una muestra de 38.027 palabras del subtrain y 12 árboles por bosque, seguida de un **refinamiento de seis configuraciones** entrenadas con las 318.900 palabras del subtrain y 40 árboles. La comparación se hizo sobre las mismas 81.167 palabras de validación.

Se analizaron especialmente el efecto de `class_weight` y `max_depth` ante el desbalance de clases. La mejor configuración de cada tarea, **dentro de las opciones exploradas**, fue:

| Hiperparámetro | Puntuación inicial | Puntuación final | Capitalización |
|---|---|---|---|
| `n_estimators` | 40 | 40 | 40 |
| `max_depth` | 20 | 20 | 20 |
| `min_samples_leaf` | 1 | 1 | 1 |
| `max_features` | `"sqrt"` | `"sqrt"` | `"sqrt"` |
| `class_weight` | `None` | `"balanced"` | `None` |

### Resultados del refinamiento (validación, antes de ajustar decisiones)

| Tarea | Candidato | Macro F1 |
|---|---|---:|
| Inicial | Sin pesos, profundidad 20 | **0,5967** |
| Inicial | Pesos balanceados, profundidad 20 | 0,5256 |
| Final | Balanceado, profundidad 12 | 0,3335 |
| Final | Balanceado, profundidad 20 | **0,3495** |
| Capitalización | Sin pesos, profundidad 20 | **0,5511** |
| Capitalización | Balanceado, profundidad 20 | 0,4373 |

**Interpretación:** ponderar automáticamente las clases no fue óptimo en las tres tareas. En puntuación inicial y capitalización ganó la opción sin pesos; en puntuación final, la balanceada. Por eso la selección se hizo **por tarea**.

## 5. Umbrales y reglas de decisión óptimos

**Importante:** solo la tarea binaria tiene un **umbral escalar**. Para las tareas multiclase no se encontró un umbral único por modelo: se optimizaron **multiplicadores de probabilidades por clase**. Son las reglas seleccionadas en validación y congeladas antes del test.

### 5.1. Puntuación inicial — umbral óptimo `0,195`

Para la clase positiva `¿`, la decisión es:

```python
pred = "¿" if proba_interrogacion >= 0.195 else ""
```

El umbral **0,195** reemplaza la decisión binaria habitual de umbral cercano a 0,5. Reducirlo permitió recuperar más interrogaciones sin el exceso de falsos positivos que se observaba en el baseline balanceado.

### 5.2. Puntuación final — multiplicadores óptimos

| Clase | Multiplicador |
|---|---:|
| Sin signo (`""`) | **1,00** |
| Coma (`,`) | **0,40** |
| Punto (`.`) | **0,40** |
| Interrogación (`?`) | **0,15** |

Para cada palabra, se multiplica la probabilidad predicha de cada clase por su factor y se selecciona la clase de mayor puntuación:

```python
pred = clases[np.argmax(probabilidades * multiplicadores)]
```

Los factores corresponden a las clases en el orden indicado por `modelo.classes_`, **no necesariamente** al orden visual de la tabla. En esta tarea la regla favorece relativamente la ausencia de signo respecto de los signos de puntuación.

### 5.3. Capitalización — multiplicadores óptimos

| Clase | Multiplicador |
|---|---:|
| `0` | **1,00** |
| `1` | **2,50** |
| `2` | **3,00** |
| `3` | **4,00** |

Se aplica la misma regla de `argmax` sobre las probabilidades multiplicadas. Esta configuración aumenta el peso relativo de las clases minoritarias frente a la clase `0`.

**Nota:** estos factores son ajustes *post-hoc* de la regla de clasificación, **no una calibración probabilística** ni cambios a la función de pérdida del Random Forest.

### Efecto del ajuste de decisiones en validación

| Tarea | Macro F1 sin ajuste | Macro F1 ajustado | Mejora absoluta |
|---|---:|---:|---:|
| Inicial | 0,5967 | **0,6935** | +0,0968 |
| Final | 0,3495 | **0,4033** | +0,0538 |
| Capitalización | 0,5511 | **0,5977** | +0,0466 |

Los ajustes se seleccionaron **maximizando macro F1 de validación**. Los resultados de validación pueden ser optimistas porque también se usó para seleccionar hiperparámetros; por eso la confirmación relevante es la evaluación final sobre test.

## 6. Evaluación final sobre el test reservado

Se reentrenaron los modelos con **400.067 palabras** y se evaluaron sobre **100.906 palabras**, sin modificar los hiperparámetros ni las reglas elegidas en validación.

### Macro F1

| Tarea | Baseline | RF optimizado, decisión original | RF optimizado, decisión ajustada | Mejora relativa vs. baseline |
|---|---:|---:|---:|---:|
| Inicial | 0,5249 | 0,5909 | **0,6767** | **+28,9 %** |
| Final | 0,3577 | 0,3498 | **0,4029** | **+12,6 %** |
| Capitalización | 0,4283 | 0,5886 | **0,6310** | **+47,3 %** |
| **Promedio** | **0,4370** | **0,5098** | **0,5702** | **+30,5 %** |

### Accuracy

| Tarea | Baseline | RF optimizado con decisión ajustada |
|---|---:|---:|
| Inicial | 84,22 % | **98,01 %** |
| Final | 63,59 % | **78,31 %** |
| Capitalización | 76,14 % | **90,12 %** |

**Accuracy conjunto:** del **44,22 %** en el baseline al **70,28 %** con los tres modelos optimizados: **+26,06 puntos porcentuales**.

El aumento del macro F1 promedio de **0,4370 a 0,5702** representa aproximadamente **+30,5 % de mejora relativa**. No significa que todas las clases individuales hayan mejorado por igual; para ese análisis deben consultarse los reportes y matrices de confusión incluidos en los resultados del experimento.

## 7. Interpretación y conclusiones

1. **El desbalance exige decisiones específicas por tarea.** `class_weight="balanced"` no fue siempre la mejor opción para macro F1; en dos tareas resultó mejor no aplicarlo.
2. **La regla de decisión importa mucho.** Los bosques estiman probabilidades, pero elegir directamente la clase de mayor probabilidad no necesariamente maximiza macro F1. Los umbrales y multiplicadores mejoraron las tres tareas en validación y mantuvieron la mejora en test.
3. **Puntuación final sigue siendo la más difícil.** Su macro F1 final (0,4029) queda por debajo del de puntuación inicial (0,6767) y capitalización (0,6310). Es razonable atribuir parte de esta dificultad a que el RF recibe embeddings estáticos por palabra y no modela explícitamente el contexto sintáctico ni la secuencia; esta es una hipótesis explicativa, no una prueba causal.
4. **El RF optimizado sirve como referencia para las redes.** Una RNN, LSTM, GRU o arquitectura bidireccional podrá utilizar el contexto de las palabras vecinas, ausente en este pipeline.

## 8. Alcance y reproducibilidad

- Estos son los mejores resultados **entre las configuraciones exploradas**, no una demostración de optimalidad global.
- El **test se reservó para la evaluación final**. No se recomienda seguir ajustando modelos en función de estos resultados; futuras búsquedas deberían hacerse con validación independiente o cruzada por fragmentos.
- El notebook asociado incluye preparación de datos, baseline, screening, refinamiento, optimización de decisiones y evaluación final.
- Para ejecutar el notebook hacen falta `embeddings_palabras.npy` y `etiquetas_palabras.csv`, que **no están embebidos** en el `.ipynb`.
- El notebook contiene código reproducible y resultados históricos; no equivale a distribuir modelos serializados ya entrenados.

**Conclusión:** la optimización del pipeline Random Forest elevó el macro F1 promedio en test de **0,4370 a 0,5702**, y el accuracy conjunto de **44,22 % a 70,28 %**. Constituye un baseline clásico optimizado para comparar con modelos secuenciales.
