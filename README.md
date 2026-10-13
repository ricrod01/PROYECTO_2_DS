# Predicciones de Commodities

Este proyecto 

## Estructura


```text
mi_proyecto/
├── app.py
├── cuadros_presentacion/
├── datos_crudos/
│   └── target_pairs.csv
├── datos_procesados/
│   ├── entrenamiento.csv
│   └── entrenamiento_etiquetas.csv
├── docs/
├── errores_modelos/
├── modelos/
│   ├── GARCH_<nombre>.joblib
│   ├── SARIMA_<nombre>.joblib
│   └── SARFIMA_<nombre>.joblib
├── notebooks/
├── scripts/
│   └── modelo_SARFIMA.py
├── README.md
└── requirements.txt
```

## Ejecución de la aplicación.

La aplicación necesita las dependencias importadas por `modelo_SARFIMA.py` en la carpeta `scripts`. Además usa los archivos .joblib de la carpeta `modelos`.

Desde la carpeta raíz del proyecto, instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Streamlit abrirá la app en tu navegador.

## Funcionalidad

- **Gráfica de series:** menú con `train.columns`; simulación con `prediccion_serie_simulado([nombre_serie])`; slider de 1 a `test.shape[0]`.
- **Gráfica de etiquetas:** menú con `etiquetas.index`; simulación con `prediccion_etiqueta_simulado(etiquetas.loc[indice])`; slider de 1 a `indices_etiquetas(lag)[1]` para la etiqueta elegida.
- Las predicciones se almacenan en caché por selección, de modo que mover los sliders no las recalcula.
- Ambas gráficas presentan las observaciones reales y las simuladas mediante Plotly.

**Nota sobre el segundo slider:** la función `indices_etiquetas(lag)` del notebook calcula `y = filas - n + lag + 1`, pero el cálculo de diferencias de `prediccion_etiqueta_simulado` genera normalmente menos puntos que ese límite para `lag > 0`. Para respetar el intervalo pedido sin generar datos ficticios, la app permite recorrer hasta `y` y dibuja como máximo los puntos realmente disponibles en ambas series. Esto se indica en la interfaz cuando corresponda.

**Importante:** el notebook adjunto no incluye los CSV, los modelos `.joblib` ni el archivo `modelo_SARFIMA.py`. Sin ellos, la app no puede calcular las predicciones reales.
