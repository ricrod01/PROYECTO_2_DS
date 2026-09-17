import numpy as np

def pesos_diferenciacion_fraccional(d, n):

    pesos = np.ones(n)

    for k in range(1, n):
        pesos[k] = -pesos[k - 1] * (d - k + 1) / k

    return pesos


def diferenciacion_fraccional(serie, d):

    serie = np.asarray(serie, dtype=float)
    pesos = pesos_diferenciacion_fraccional(d=d, n=len(serie))
    serie_diff = np.convolve(serie, pesos, mode="full")[:len(serie)]

    return serie_diff


class ModeloSARFIMA:

    modelo: object
    d_fraccional: float
    historia: np.ndarray
    periodo_estacional: int

    def predict(self, n_periods):

        # Predicción sobre la serie diferenciada
        pred_diff = np.asarray(self.modelo.predict(n_periods=n_periods), dtype=float)

        historia = list(np.asarray(self.historia, dtype=float))
        n_train = len(historia)

        # Se calculan suficientes pesos también para las observaciones futuras
        pesos = pesos_diferenciacion_fraccional(d=self.d_fraccional, n=n_train + n_periods)

        predicciones = []

        # Se reconstruye la serie original
        for z_t in pred_diff:

            t = len(historia)
            acumulado = 0.0

            for k in range(1, t + 1):
                acumulado += (pesos[k] * historia[t - k])

            y_t = z_t - acumulado
            historia.append(y_t)
            predicciones.append(y_t)

        return np.asarray(predicciones)