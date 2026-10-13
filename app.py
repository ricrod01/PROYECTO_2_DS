"""Aplicación Streamlit basada en METRICA_MODELAJE(1).ipynb.

Colocar este archivo en la raíz del proyecto, junto a las carpetas
`datos_procesados`, `datos_crudos`, `modelos` y `scripts`.
Ejecutar con: streamlit run app.py
"""

from pathlib import Path
import inspect
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------
# 1. Configuración de la aplicación y rutas del proyecto
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Predicciones simuladas",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "datos_procesados" / "entrenamiento.csv"
RUTA_ETIQUETAS = BASE_DIR / "datos_procesados" / "entrenamiento_etiquetas.csv"
RUTA_CATALOGO = BASE_DIR / "datos_crudos" / "target_pairs.csv"
RUTA_MODELOS = BASE_DIR / "modelos"
RUTA_SCRIPT = BASE_DIR / "scripts" / "modelo_SARFIMA.py"

st.title("📈 Panel de predicciones simuladas")
st.caption("Comparación interactiva de observaciones reales y predicciones simuladas.")

faltantes = [
    ruta for ruta in (RUTA_DATOS, RUTA_ETIQUETAS, RUTA_CATALOGO, RUTA_SCRIPT)
    if not ruta.is_file()
]
if not RUTA_MODELOS.is_dir():
    faltantes.append(RUTA_MODELOS)

if faltantes:
    st.error("Faltan archivos necesarios para ejecutar las predicciones del notebook.")
    st.code("\n".join(str(ruta) for ruta in faltantes), language="text")
    st.info("Consulta el README para la estructura de carpetas esperada.")
    st.stop()

# Los modelos serializados con joblib pueden depender de esta clase propia.
sys.path.insert(0, str(RUTA_SCRIPT.parent))
try:
    from modelo_SARFIMA import pesos_diferenciacion_fraccional
except Exception as exc:
    st.error(f"No se pudo importar modelo_SARFIMA.py: {exc}")
    st.stop()


# ---------------------------------------------------------------
# 2. Funciones originales de búsqueda y simulación del notebook
# ---------------------------------------------------------------
def buscar_modelo(nombres):
    """Devuelve archivos y códigos de tipo para una lista de series.

    Tipos: SARIMA=1, SARFIMA=2 y GARCH=3.
    Se conserva la prioridad de búsqueda del notebook: GARCH, SARIMA, SARFIMA.
    """
    algoritmos = ["GARCH", "SARIMA", "SARFIMA"]
    archivos = []
    tipos = []

    for nombre in nombres:
        for algoritmo in algoritmos:
            nombre_archivo = f"{algoritmo}_{nombre}.joblib"
            if (RUTA_MODELOS / nombre_archivo).is_file():
                archivos.append(nombre_archivo)
                if algoritmo == "SARIMA":
                    tipos.append(1)
                elif algoritmo == "SARFIMA":
                    tipos.append(2)
                else:
                    tipos.append(3)
    return archivos, tipos


def cargar_modelo(nombre_archivo):
    """Carga exclusivamente modelos propios, previamente guardados en modelos/."""
    return joblib.load(RUTA_MODELOS / nombre_archivo)


def indices_etiquetas(lag):
    """Regresa (índice inicial de prueba, máximo teórico) del notebook."""
    x = n - 1
    y = filas - n + lag + 1
    return x, y


def simular_statsmodels(resultado, pasos, semilla=0):
    """Simula con semilla reproducible en statsmodels 0.14 y 0.15+.

    statsmodels 0.15 utiliza `rng`; 0.14 utiliza `random_state`.
    Se envía solo el parámetro compatible, nunca ambos.
    """
    parametros = inspect.signature(resultado.simulate).parameters
    nombre_semilla = "rng" if "rng" in parametros else "random_state"

    return resultado.simulate(
        nsimulations=pasos,
        anchor="end",
        **{nombre_semilla: semilla},
    )


def predecir_simulado(modelo, tipo, pasos, semilla=0):
    """Simula SARIMA/SARFIMA/GARCH, conservando semilla=0 del notebook.

    Parámetros:
      - tipo: 1 (SARIMA), 2 (SARFIMA) o 3 (GARCH).
      - pasos: número de observaciones a simular.
      - semilla: estado inicial del generador aleatorio (reproducibilidad).
    """
    if tipo == 1:  # SARIMA
        resultado = modelo.arima_res_ if hasattr(modelo, "arima_res_") else modelo
        prediccion = simular_statsmodels(resultado, pasos, semilla)
    elif tipo == 2:  # SARFIMA
        resultado = (
            modelo.modelo.arima_res_
            if hasattr(modelo.modelo, "arima_res_")
            else modelo.modelo
        )
        pred_diff = np.asarray(
            simular_statsmodels(resultado, pasos, semilla)
        ).flatten()
        historia = list(np.asarray(modelo.historia, dtype=float))
        pesos = pesos_diferenciacion_fraccional(
            modelo.d_fraccional, len(historia) + pasos
        )
        prediccion = []
        for z_t in pred_diff:
            t = len(historia)
            y_t = z_t - np.dot(pesos[1 : t + 1], historia[::-1])
            historia.append(y_t)
            prediccion.append(y_t)
    elif tipo == 3:  # GARCH
        rng = np.random.default_rng(semilla)
        resultado = modelo.forecast(horizon=pasos, reindex=False)
        media = resultado.mean.iloc[-1].to_numpy()
        desviacion = np.sqrt(resultado.variance.iloc[-1].to_numpy())
        prediccion = media + rng.normal(0, 1, pasos) * desviacion
    else:
        raise ValueError("Tipo inválido. Usa 1, 2 o 3.")

    return np.asarray(prediccion).flatten()


def prediccion_serie_simulado(nombre_serie):
    """Recibe una lista de nombres, como en el notebook: [nombre_serie]."""
    archivos, tipos = buscar_modelo(nombre_serie)
    if not archivos:
        raise FileNotFoundError(f"No se encontró un modelo para {nombre_serie!r}.")
    modelo = cargar_modelo(archivos[0])
    y_real = df[nombre_serie][n:]
    y_pred = predecir_simulado(modelo, tipos[0], filas - n)
    return y_real, y_pred


def prediccion_etiqueta_simulado(serie_etiqueta):
    """Recibe una fila de etiquetas; mantiene la lógica del notebook."""
    target = serie_etiqueta["target"]
    lag = int(serie_etiqueta["lag"])
    archivos = serie_etiqueta["archivos"]
    tipos = serie_etiqueta["tipos"]
    indice_prueba = int(serie_etiqueta["indice_prueba"])
    pasos = int(serie_etiqueta["pasos"]) + 1
    cantidad_modelos = len(archivos)

    if cantidad_modelos not in (1, 2):
        raise ValueError(
            f"La etiqueta {target!r} necesita 1 o 2 modelos; "
            f"se encontraron {cantidad_modelos}."
        )

    modelo_1 = cargar_modelo(archivos[0])
    if cantidad_modelos == 2:
        modelo_2 = cargar_modelo(archivos[1])

    y_real = df_etiquetas[target][indice_prueba:]

    y_1 = predecir_simulado(modelo_1, tipos[0], pasos + lag - 1)
    y_1_diff = (y_1[lag:] - y_1[:-lag])[:-lag]
    if cantidad_modelos == 2:
        y_2 = predecir_simulado(modelo_2, tipos[1], pasos + lag - 1)
        y_2_diff = (y_2[lag:] - y_2[:-lag])[:-lag]
        y_pred = y_1_diff - y_2_diff
    else:
        y_pred = y_1_diff

    return y_real, y_pred


# ---------------------------------------------------------------
# 3. Lectura y preparación de datos del notebook
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def leer_archivos():
    datos = pd.read_csv(RUTA_DATOS, index_col=0)
    datos_etiquetas = pd.read_csv(RUTA_ETIQUETAS, index_col=0)
    catalogo = pd.read_csv(RUTA_CATALOGO)
    return datos, datos_etiquetas, catalogo


try:
    df, df_etiquetas, etiquetas = leer_archivos()
except Exception as exc:
    st.error(f"No se pudieron leer los CSV: {exc}")
    st.stop()

filas = df.shape[0]
n = int(filas * 0.9)
train = df.iloc[:n]
test = df.iloc[n:]

if train.empty or test.empty or etiquetas.empty:
    st.error("Los conjuntos de entrenamiento, prueba y etiquetas deben contener datos.")
    st.stop()

try:
    etiquetas["etiquetas"] = [x.split(" - ") for x in etiquetas["pair"]]
    etiquetas[["archivos", "tipos"]] = pd.DataFrame(
        etiquetas["etiquetas"].apply(buscar_modelo).tolist(), index=etiquetas.index
    )
    etiquetas[["indice_prueba", "pasos"]] = pd.DataFrame(
        etiquetas["lag"].apply(indices_etiquetas).tolist(), index=etiquetas.index
    )
except Exception as exc:
    st.error(f"No fue posible preparar el catálogo de etiquetas: {exc}")
    st.stop()


# Cachear la simulación, no el deslizador: moverlo solo cambia los puntos dibujados.
@st.cache_data(show_spinner="Calculando la simulación de la serie...")
def prediccion_serie_cache(nombre_serie):
    reales, simulados = prediccion_serie_simulado([nombre_serie])
    return np.asarray(reales, dtype=float).reshape(-1), np.asarray(simulados, dtype=float).reshape(-1)


@st.cache_data(show_spinner="Calculando la simulación de la etiqueta...")
def prediccion_etiqueta_cache(indice):
    reales, simulados = prediccion_etiqueta_simulado(etiquetas.loc[indice])
    return np.asarray(reales, dtype=float).reshape(-1), np.asarray(simulados, dtype=float).reshape(-1)


def crear_grafica(reales, simulados, cantidad, titulo):
    """Muestra los primeros `cantidad` datos comunes a ambas secuencias."""
    visibles = min(cantidad, len(reales), len(simulados))
    periodos = np.arange(visibles)

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=periodos,
            y=reales[:visibles],
            mode="lines",
            name="Valores reales",
            line=dict(color="#2563eb", width=2),
        )
    )
    figura.add_trace(
        go.Scatter(
            x=periodos,
            y=simulados[:visibles],
            mode="lines",
            name="Predicciones",
            line=dict(color="#e11d48", width=2),
        )
    )
    figura.update_layout(
        title=dict(text=titulo, font=dict(size=16)),
        xaxis_title="Período de prueba",
        yaxis_title="Valor",
        hovermode="x unified",
        height=440,
        margin=dict(l=30, r=20, t=65, b=40),
        legend=dict(orientation="h", y=1.12, x=0),
        template="plotly_white",
    )
    figura.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    figura.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return figura, visibles


def explicar_formacion_etiqueta(fila_etiqueta):
    """Construye la explicación de `target` a partir de `pair` y `lag`.

    La regla reproduce el cálculo de prediccion_etiqueta_simulado:
    primero se toma la diferencia con un rezago `lag` por serie y, cuando
    existe un par, se resta la diferencia de la segunda serie.
    """
    target = str(fila_etiqueta["target"])
    pair = str(fila_etiqueta["pair"])
    lag = int(fila_etiqueta["lag"])
    series = [serie.strip() for serie in pair.split(" - ")]

    if len(series) == 1:
        serie = series[0]
        regla = f"{serie}(t) − {serie}(t−{lag})"
        explicacion = "Variación de la serie respecto a su valor rezagado."
    elif len(series) == 2:
        serie_1, serie_2 = series
        regla = (
            f"[{serie_1}(t) − {serie_1}(t−{lag})] "
            f"− [{serie_2}(t) − {serie_2}(t−{lag})]"
        )
        explicacion = "Variación de la primera serie menos la variación de la segunda."
    else:
        # El notebook espera 1 o 2 series; mostrar los datos sin inventar una fórmula.
        regla = "No disponible para este formato de pair."
        explicacion = "El catálogo contiene una composición no contemplada por el notebook."

    return (
        f"**Cómo se forma la etiqueta `{target}`**\n\n"
        f"**Pair:** `{pair}`  \n"
        f"**Lag:** `{lag}` período(s)  \n"
        f"**Regla:** {explicacion}  \n"
        f"**Fórmula:** `{regla}`"
    )


# ---------------------------------------------------------------
# 4. Panel: dos gráficas independientes con sus selectores y sliders
# ---------------------------------------------------------------
col_series, col_etiquetas = st.columns(2, gap="large")

with col_series:
    st.subheader("Serie temporal")
    serie_seleccionada = st.selectbox(
        "Seleccionar serie",
        options=train.columns.tolist(),
        key="selector_serie",
    )
    cantidad_serie = st.slider(
        "Cantidad de datos visibles",
        min_value=1,
        max_value=int(test.shape[0]),
        value=int(test.shape[0]),
        step=1,
        key="slider_serie",
    )

    try:
        real_serie, pred_serie = prediccion_serie_cache(serie_seleccionada)
        grafica, visibles = crear_grafica(
            real_serie,
            pred_serie,
            cantidad_serie,
            f"Serie: {serie_seleccionada}",
        )
        st.plotly_chart(grafica, use_container_width=True, config={"displaylogo": False})
        st.caption(f"Observaciones mostradas: {visibles} de {test.shape[0]}")
    except Exception as exc:
        st.error(f"No se pudo generar la gráfica de la serie: {exc}")

with col_etiquetas:
    st.subheader("Etiqueta")
    etiqueta_seleccionada = st.selectbox(
        "Seleccionar etiqueta (índice de etiquetas)",
        options=etiquetas.index.tolist(),
        format_func=str,
        key="selector_etiqueta",
    )
    fila_etiqueta = etiquetas.loc[etiqueta_seleccionada]
    lag = int(fila_etiqueta["lag"])
    _, limite_etiqueta = indices_etiquetas(lag)

    cantidad_etiqueta = st.slider(
        "Cantidad de datos visibles ",
        min_value=1,
        max_value=int(limite_etiqueta),
        value=int(limite_etiqueta),
        step=1,
        key=f"slider_etiqueta_{etiqueta_seleccionada}",
    )

    try:
        real_etiqueta, pred_etiqueta = prediccion_etiqueta_cache(etiqueta_seleccionada)
        grafica, visibles = crear_grafica(
            real_etiqueta,
            pred_etiqueta,
            cantidad_etiqueta,
            f"Etiqueta: {etiqueta_seleccionada} — {fila_etiqueta['target']}",
        )
        st.plotly_chart(grafica, use_container_width=True, config={"displaylogo": False})
        st.caption(f"Observaciones mostradas: {visibles} de {limite_etiqueta} seleccionables · lag = {lag}")
    except Exception as exc:
        st.error(f"No se pudo generar la gráfica de la etiqueta: {exc}")

    # Mantener el cuadro informativo siempre visible, sin mensajes del slider.
    st.info(explicar_formacion_etiqueta(fila_etiqueta), icon="ℹ️")
