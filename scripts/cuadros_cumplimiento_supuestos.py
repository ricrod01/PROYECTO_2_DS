"""Genera cuadros de supuestos y gráficas del mejor y peor modelo.

Los conteos provienen de los notebooks Supuestos_SARIMA.ipynb,
Supuestos_SARFIMA.ipynb y Supuestos_GARCH.ipynb. Los denominadores son
distintos porque cada notebook evaluó un subconjunto diferente de modelos.
"""

from __future__ import annotations

import argparse
import __main__
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kstest, ttest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from modelo_SARFIMA import (
    ModeloSARFIMA,
    diferenciacion_fraccional,
    pesos_diferenciacion_fraccional,
)


DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
ALFA = 0.05

CRITERIOS = {
    "SARIMA": [
        ("Ausencia de autocorrelación", "sin_autocorrelacion"),
        ("Normalidad de residuos", "normalidad"),
        ("Homocedasticidad", "homocedasticidad"),
        ("Media cero", None),
        ("Estacionariedad y estabilidad", None),
    ],
    "SARFIMA": [
        ("Media cero", "media_cero"),
        ("Residuos como ruido blanco", "ruido_blanco"),
        ("Normalidad", "normalidad"),
        ("Homocedasticidad", "homocedasticidad"),
        ("Estacionariedad (ADF y KPSS)", "estacionaria_diff"),
        ("Estabilidad de raices", "raices_estables"),
    ],
    "GARCH": [
        ("Media cero", "media_cero"),
        ("Sin autocorrelación lineal", "sin_autocorr_lineal"),
        ("Sin autocorrelación en cuadrados", "sin_autocorr_cuadrados"),
        ("Sin efecto ARCH remanente", "sin_arch"),
        ("Distribución adecuada", "distribucion_adecuada"),
        ("Convergencia", "convergencia"),
        ("Persistencia menor que uno", "varianza_estacionaria"),
    ],
}


def cargar_modelos(patron: str) -> dict[str, object]:
    """Carga todos los modelos que coinciden con un patrón."""
    rutas = sorted((DIRECTORIO_PROYECTO / "modelos").glob(patron))
    if not rutas:
        raise FileNotFoundError(f"No se encontraron modelos con el patrón {patron}")
    return {
        ruta.stem.split("_", 1)[1]: joblib.load(ruta)
        for ruta in rutas
    }


def evaluar_sarima() -> pd.DataFrame:
    """Evalúa los residuos de todos los modelos SARIMA actuales."""
    filas = []
    for nombre, modelo in cargar_modelos("SARIMA_*.joblib").items():
        try:
            residuos = pd.Series(modelo.resid(), dtype=float).dropna()
            lb = acorr_ljungbox(
                residuos, lags=[12, 24, 36], return_df=True
            )
            jb_p = float(jarque_bera(residuos).pvalue)
            arch_p = float(het_arch(residuos)[1])
            filas.append(
                {
                    "modelo": nombre,
                    "n_residuos": len(residuos),
                    "media_residuos": residuos.mean(),
                    "std_residuos": residuos.std(),
                    "jb_p": jb_p,
                    "arch_p": arch_p,
                    "ljung_box_p_min": lb["lb_pvalue"].min(),
                    "sin_autocorrelacion": bool(
                        (lb["lb_pvalue"] > ALFA).all()
                    ),
                    "normalidad": jb_p > ALFA,
                    "homocedasticidad": arch_p > ALFA,
                    "error_diagnostico": None,
                }
            )
        except Exception as error:
            filas.append({"modelo": nombre, "error_diagnostico": str(error)})
    return pd.DataFrame(filas)


def minimo_modulo_raices(funcion_raices) -> float:
    """Devuelve el menor módulo o infinito cuando no existen raíces."""
    raices = np.asarray(funcion_raices(), dtype=complex)
    return float(np.min(np.abs(raices))) if raices.size else float("inf")


def evaluar_sarfima() -> pd.DataFrame:
    """Aplica los diagnósticos originales a todos los modelos SARFIMA."""
    # Algunos modelos se serializaron cuando estas definiciones estaban en __main__.
    __main__.ModeloSARFIMA = ModeloSARFIMA
    __main__.diferenciacion_fraccional = diferenciacion_fraccional
    __main__.pesos_diferenciacion_fraccional = pesos_diferenciacion_fraccional

    filas = []
    for nombre, modelo in cargar_modelos("SARFIMA_*.joblib").items():
        try:
            arima = modelo.modelo
            p, _, q = arima.order
            p_est, _, q_est, periodo = arima.seasonal_order
            periodo = periodo or modelo.periodo_estacional or 1
            grados_libertad = p + q + p_est + q_est
            burn_in = max(
                10,
                2 * periodo,
                p + q + p_est * periodo + q_est * periodo,
            )

            residuos = np.asarray(arima.resid(), dtype=float)[burn_in:]
            residuos = residuos[np.isfinite(residuos)]
            serie_diff = diferenciacion_fraccional(
                modelo.historia, modelo.d_fraccional
            )[burn_in:]
            serie_diff = serie_diff[np.isfinite(serie_diff)]

            lags = [
                lag
                for lag in (periodo, 2 * periodo, 3 * periodo)
                if grados_libertad < lag < len(residuos)
            ]
            if not lags:
                lags = [
                    min(max(grados_libertad + 1, 1), len(residuos) - 1)
                ]

            lb = acorr_ljungbox(
                residuos,
                lags=lags,
                model_df=grados_libertad,
                return_df=True,
            )
            media_p = float(ttest_1samp(residuos, popmean=0.0).pvalue)
            jb_p = float(jarque_bera(residuos).pvalue)
            arch_lags = min(3 * periodo, int(np.sqrt(len(residuos))))
            arch_p = float(het_arch(residuos, nlags=arch_lags)[1])
            adf_p = float(adfuller(serie_diff, autolag="AIC")[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p = float(kpss(serie_diff, regression="c", nlags="auto")[1])
            raiz_ar = minimo_modulo_raices(arima.arroots)
            raiz_ma = minimo_modulo_raices(arima.maroots)

            filas.append(
                {
                    "modelo": nombre,
                    "n_residuos": len(residuos),
                    "burn_in": burn_in,
                    "media_p": media_p,
                    "ljung_box_p_min": lb["lb_pvalue"].min(),
                    "jb_p": jb_p,
                    "arch_p": arch_p,
                    "adf_p": adf_p,
                    "kpss_p": kpss_p,
                    "raiz_ar_min": raiz_ar,
                    "raiz_ma_min": raiz_ma,
                    "media_cero": media_p > ALFA,
                    "ruido_blanco": lb["lb_pvalue"].min() > ALFA,
                    "normalidad": jb_p > ALFA,
                    "homocedasticidad": arch_p > ALFA,
                    "estacionaria_diff": adf_p < ALFA and kpss_p > ALFA,
                    "raices_estables": raiz_ar > 1 and raiz_ma > 1,
                    "error_diagnostico": None,
                }
            )
        except Exception as error:
            filas.append({"modelo": nombre, "error_diagnostico": str(error)})
    return pd.DataFrame(filas)


def evaluar_garch() -> pd.DataFrame:
    """Aplica los diagnósticos originales a todos los modelos GARCH."""
    filas = []
    for nombre, modelo in cargar_modelos("GARCH_*.joblib").items():
        try:
            lags_ar = modelo.model.lags
            orden_ar = (
                int(np.max(lags_ar))
                if lags_ar is not None and np.asarray(lags_ar).size
                else 0
            )
            p = modelo.model.volatility.p
            q = modelo.model.volatility.q
            burn_in = max(10, orden_ar + p + q)

            residuos = np.asarray(modelo.resid, dtype=float)[burn_in:]
            residuos_std = np.asarray(modelo.std_resid, dtype=float)[burn_in:]
            volatilidad = np.asarray(
                modelo.conditional_volatility, dtype=float
            )[burn_in:]
            validos = (
                np.isfinite(residuos)
                & np.isfinite(residuos_std)
                & np.isfinite(volatilidad)
            )
            residuos_std = residuos_std[validos]
            volatilidad = volatilidad[validos]

            lags_media = [
                lag
                for lag in (10, 20, 30)
                if orden_ar < lag < len(residuos_std)
            ]
            lags_varianza = [
                lag
                for lag in (10, 20, 30)
                if p + q < lag < len(residuos_std)
            ]
            lb = acorr_ljungbox(
                residuos_std,
                lags=lags_media,
                model_df=orden_ar,
                return_df=True,
            )
            lb_cuadrados = acorr_ljungbox(
                residuos_std**2,
                lags=lags_varianza,
                model_df=p + q,
                return_df=True,
            )
            media_p = float(ttest_1samp(residuos_std, popmean=0.0).pvalue)
            arch_lags = min(20, int(np.sqrt(len(residuos_std))))
            arch_p = float(het_arch(residuos_std, nlags=arch_lags)[1])

            distribucion = modelo.model.distribution
            parametros_dist = np.asarray(
                [modelo.params[nombre] for nombre in distribucion.parameter_names()]
            )
            pit = distribucion.cdf(residuos_std, parametros_dist)
            pit = np.clip(pit, np.finfo(float).eps, 1 - np.finfo(float).eps)
            pit_p = float(kstest(pit, "uniform").pvalue)
            persistencia = float(
                modelo.params[
                    modelo.params.index.str.startswith(("alpha[", "beta["))
                ].sum()
            )

            filas.append(
                {
                    "modelo": nombre,
                    "n_residuos": len(residuos_std),
                    "burn_in": burn_in,
                    "media_p": media_p,
                    "ljung_box_p_min": lb["lb_pvalue"].min(),
                    "ljung_box_cuadrados_p_min": lb_cuadrados[
                        "lb_pvalue"
                    ].min(),
                    "arch_p": arch_p,
                    "jb_p": float(jarque_bera(residuos_std).pvalue),
                    "pit_ks_p": pit_p,
                    "persistencia": persistencia,
                    "convergence_flag": modelo.convergence_flag,
                    "varianzas_positivas": bool((volatilidad > 0).all()),
                    "media_cero": media_p > ALFA,
                    "sin_autocorr_lineal": lb["lb_pvalue"].min() > ALFA,
                    "sin_autocorr_cuadrados": (
                        lb_cuadrados["lb_pvalue"].min() > ALFA
                    ),
                    "sin_arch": arch_p > ALFA,
                    "distribucion_adecuada": pit_p > ALFA,
                    "convergencia": modelo.convergence_flag == 0,
                    "varianza_estacionaria": persistencia < 1,
                    "error_diagnostico": None,
                }
            )
        except Exception as error:
            filas.append({"modelo": nombre, "error_diagnostico": str(error)})
    return pd.DataFrame(filas)


def evaluar_supuestos() -> dict[str, pd.DataFrame]:
    """Ejecuta los diagnósticos de las tres familias."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You've deserialized an ARIMA.*")
        return {
            "SARIMA": evaluar_sarima(),
            "SARFIMA": evaluar_sarfima(),
            "GARCH": evaluar_garch(),
        }


def crear_cuadro(modelo: str, diagnosticos: pd.DataFrame) -> pd.DataFrame:
    """Construye el cuadro de cumplimiento para una familia de modelos."""
    filas = []
    for supuesto, columna in CRITERIOS[modelo]:
        if columna is None or columna not in diagnosticos:
            evaluados = cumplen = no_cumplen = None
        else:
            valores = diagnosticos[columna].dropna().astype(bool)
            evaluados = len(valores)
            cumplen = int(valores.sum())
            no_cumplen = evaluados - cumplen
        porcentaje = None if evaluados is None else 100 * cumplen / evaluados
        filas.append(
            {
                "Supuesto": supuesto,
                "Evaluados": evaluados,
                "Cumplen": cumplen,
                "No cumplen": no_cumplen,
                "Cumplimiento (%)": porcentaje,
            }
        )
    return pd.DataFrame(filas)


def formatear_cuadro(cuadro: pd.DataFrame) -> pd.DataFrame:
    """Prepara los valores para mostrarlos sin alterar los datos originales."""
    salida = cuadro.copy()
    for columna in ("Evaluados", "Cumplen", "No cumplen"):
        salida[columna] = salida[columna].map(
            lambda valor: "N/E" if pd.isna(valor) else str(int(valor))
        )
    salida["Cumplimiento (%)"] = salida["Cumplimiento (%)"].map(
        lambda valor: "N/E" if pd.isna(valor) else f"{valor:.2f}%"
    )
    return salida


def color_cumplimiento(valor: float | None) -> str:
    """Asigna un color según el porcentaje de cumplimiento."""
    if valor is None or pd.isna(valor):
        return "#D9D9D9"
    if valor >= 80:
        return "#C6E0B4"
    if valor >= 50:
        return "#FFE699"
    return "#F4B7B2"


def graficar_cuadro(modelo: str, cuadro: pd.DataFrame):
    """Crea una figura con el cuadro de una familia de modelos."""
    cuadro_visible = formatear_cuadro(cuadro)
    alto = max(3.2, 1.0 + 0.55 * len(cuadro_visible))
    figura, eje = plt.subplots(figsize=(12, alto))
    eje.axis("off")

    tabla = eje.table(
        cellText=cuadro_visible.values,
        colLabels=cuadro_visible.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.42, 0.13, 0.13, 0.14, 0.18],
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    for columna in range(len(cuadro_visible.columns)):
        celda = tabla[(0, columna)]
        celda.set_facecolor("#1F4E78")
        celda.set_text_props(color="white", weight="bold")

    columna_porcentaje = cuadro_visible.columns.get_loc("Cumplimiento (%)")
    for fila, porcentaje in enumerate(cuadro["Cumplimiento (%)"], start=1):
        tabla[(fila, columna_porcentaje)].set_facecolor(
            color_cumplimiento(porcentaje)
        )
        tabla[(fila, 0)].set_text_props(ha="left")

    eje.set_title(
        f"Cumplimiento de supuestos - {modelo}",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    figura.text(
        0.5,
        0.02,
        "N/E: supuesto no evaluado formalmente en el notebook correspondiente.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figura.tight_layout()
    return figura


def obtener_predicciones_extremas() -> dict[str, dict]:
    """Identifica los modelos extremos y reproduce sus pronósticos de prueba."""
    datos = pd.read_csv(DIRECTORIO_PROYECTO / "entrenamiento.csv")
    metricas = pd.read_csv(
        DIRECTORIO_PROYECTO / "mejores_modelos.csv", index_col=0
    )
    metricas = metricas[
        metricas["mejor_modelo"].isin(("SARIMA", "SARFIMA", "GARCH"))
    ].copy()
    metricas["error"] = pd.to_numeric(metricas["error"], errors="coerce")
    metricas = (
        metricas.dropna(subset=["error"])
        .reset_index(names="serie")
        .drop_duplicates(subset="serie", keep="last")
    )

    if metricas.empty:
        raise ValueError("No se encontraron resultados válidos en mejores_modelos.csv")

    extremos = {
        "Mejor modelo": metricas.loc[metricas["error"].idxmin()],
        "Peor modelo": metricas.loc[metricas["error"].idxmax()],
    }
    colores = {"Mejor modelo": "#2F75B5", "Peor modelo": "#C00000"}
    corte = int(len(datos) * 0.9)
    resultados = {}

    for etiqueta, registro in extremos.items():
        serie = registro["serie"]
        familia = registro["mejor_modelo"]
        archivo = DIRECTORIO_PROYECTO / "modelos" / f"{familia}_{serie}.joblib"
        if not archivo.exists():
            raise FileNotFoundError(f"No se encontró el modelo seleccionado: {archivo}")

        modelo = joblib.load(archivo)
        real = datos[serie].iloc[corte:].to_numpy(dtype=float)

        if familia == "GARCH":
            prediccion = (
                modelo.forecast(horizon=len(real), reindex=False)
                .mean.iloc[-1]
                .to_numpy(dtype=float)
            )
        else:
            prediccion = np.asarray(
                modelo.predict(n_periods=len(real)), dtype=float
            )

        rmse = float(np.sqrt(np.mean((real - prediccion) ** 2)))
        rmse_reportado = float(registro["error"])
        if not np.isclose(rmse, rmse_reportado, rtol=1e-7, atol=1e-10):
            raise ValueError(
                f"El RMSE reproducido para {serie} ({rmse:.10f}) no coincide "
                f"con el reportado ({rmse_reportado:.10f})."
            )

        resultados[etiqueta] = {
            "serie": serie,
            "familia": familia,
            "archivo": archivo.name,
            "color": colores[etiqueta],
            "rmse_reportado": rmse_reportado,
            "date_id": datos["date_id"].iloc[corte:].to_numpy(),
            "real": real,
            "prediccion": prediccion,
            "rmse": rmse,
        }

    return resultados


def graficar_modelos_extremos(resultados: dict[str, dict]):
    """Compara en un solo eje el mejor y el peor modelo."""
    figura, eje = plt.subplots(figsize=(14, 7))

    for etiqueta, resultado in resultados.items():
        color_real = "#17365D" if etiqueta == "Mejor modelo" else "#7F1D1D"
        nombre_corto = resultado["serie"].replace("US_Stock_", "").replace(
            "_adj_close", ""
        )
        eje.plot(
            resultado["date_id"],
            resultado["real"],
            color=color_real,
            linewidth=2.0,
            label=f"{nombre_corto} real ({etiqueta.lower()})",
        )
        eje.plot(
            resultado["date_id"],
            resultado["prediccion"],
            color=resultado["color"],
            linewidth=2.2,
            linestyle="--",
            label=(
                f"{nombre_corto} predicción {resultado['familia']} "
                f"(RMSE={resultado['rmse']:.5f})"
            ),
        )

    eje.set_title(
        "Mejor y peor modelo en el mismo gráfico",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    eje.set_xlabel("date_id (conjunto de prueba)")
    eje.set_ylabel("Valor logarítmico")
    eje.grid(alpha=0.25)
    eje.legend(loc="lower left", frameon=True, framealpha=0.95)
    figura.tight_layout()
    return figura


def mostrar_en_consola(cuadros: dict[str, pd.DataFrame]) -> None:
    """Imprime todos los cuadros en un formato legible."""
    for modelo, cuadro in cuadros.items():
        print(f"\n{'=' * 20} {modelo} {'=' * 20}")
        print(formatear_cuadro(cuadro).to_string(index=False))


def guardar_resultados(
    cuadros: dict[str, pd.DataFrame],
    diagnosticos: dict[str, pd.DataFrame],
    extremos: dict[str, dict],
    directorio: Path,
) -> None:
    """Exporta los cuadros y la comparación de modelos como CSV y PNG."""
    directorio.mkdir(parents=True, exist_ok=True)
    for modelo, cuadro in cuadros.items():
        nombre = modelo.lower()
        cuadro.to_csv(
            directorio / f"cumplimiento_{nombre}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        figura = graficar_cuadro(modelo, cuadro)
        figura.savefig(
            directorio / f"cumplimiento_{nombre}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figura)
        diagnosticos[modelo].to_csv(
            directorio / f"diagnostico_detallado_{nombre}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    figura_extremos = graficar_modelos_extremos(extremos)
    figura_extremos.savefig(
        directorio / "comparacion_mejor_peor_mismo_plot.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura_extremos)

    predicciones = []
    for etiqueta, resultado in extremos.items():
        predicciones.append(
            pd.DataFrame(
                {
                    "clasificacion": etiqueta,
                    "modelo": resultado["familia"],
                    "serie": resultado["serie"],
                    "date_id": resultado["date_id"],
                    "real": resultado["real"],
                    "prediccion": resultado["prediccion"],
                    "rmse": resultado["rmse"],
                }
            )
        )
    pd.concat(predicciones, ignore_index=True).to_csv(
        directorio / "predicciones_mejor_peor_modelo.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera cuadros de cumplimiento para SARIMA, SARFIMA y GARCH."
    )
    parser.add_argument(
        "--guardar",
        type=Path,
        metavar="DIRECTORIO",
        help="Directorio opcional donde se guardarán los cuadros en CSV y PNG.",
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Abre las figuras de los cuadros en ventanas de Matplotlib.",
    )
    args = parser.parse_args()

    diagnosticos = evaluar_supuestos()
    cuadros = {
        modelo: crear_cuadro(modelo, diagnosticos[modelo])
        for modelo in CRITERIOS
    }
    extremos = obtener_predicciones_extremas()
    mostrar_en_consola(cuadros)
    for modelo, detalle in diagnosticos.items():
        errores = detalle["error_diagnostico"].notna().sum()
        print(
            f"{modelo}: {len(detalle) - errores} modelos diagnosticados, "
            f"{errores} errores."
        )
    for etiqueta, resultado in extremos.items():
        print(
            f"{etiqueta}: {resultado['familia']} - {resultado['serie']} "
            f"(RMSE: {resultado['rmse']:.7f})"
        )

    if args.guardar:
        guardar_resultados(cuadros, diagnosticos, extremos, args.guardar)
        print(f"\nArchivos guardados en: {args.guardar.resolve()}")

    if args.mostrar:
        for modelo, cuadro in cuadros.items():
            graficar_cuadro(modelo, cuadro)
        graficar_modelos_extremos(extremos)
        plt.show()


if __name__ == "__main__":
    main()
