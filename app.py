"""Predicción del cumplimiento de tareas académicas - Regresión lineal múltiple con ecuación normal (versión Streamlit)."""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

VARS = ["horas_estudio", "tareas", "dificultad", "horas_redes", "horas_planificacion"]   # X1..X5
ETIQ = ["X1  Horas de estudio por semana", "X2  Tareas asignadas", "X3  Dificultad promedio (1-5)",
        "X4  Horas en redes / entretenimiento", "X5  Horas de planificación"]
Y = "cumplimiento"                                                                        # variable a predecir
CSV_POR_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_estudiantes.csv")


def metricas(real, pred):                      # MAE, RMSE y R²
    e = real - pred
    return np.mean(abs(e)), np.sqrt(np.mean(e ** 2)), 1 - (e ** 2).sum() / ((real - real.mean()) ** 2).sum()


def predecir_df(beta, t):
    return np.clip(np.c_[np.ones(len(t)), t[VARS].to_numpy(float)] @ beta, 0, 100).ravel()


@st.cache_data(show_spinner=False)
def entrenar(df):
    d = df.sample(frac=1, random_state=7)                      # mezclar los registros
    n = int(len(d) * 0.8); tr, te = d[:n], d[n:]               # 80 % entrena, 20 % prueba
    X = np.c_[np.ones(n), tr[VARS].to_numpy(float)]            # matriz X (columna de 1 = intercepto β0)
    y = tr[Y].to_numpy(float).reshape(-1, 1)                   # vector Y (cumplimiento real)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y                    # ECUACIÓN NORMAL: β = (XᵀX)⁻¹ Xᵀ Y
    return dict(X=X, y=y, beta=beta.ravel(), tr=tr, te=te,
                m_tr=metricas(tr[Y].to_numpy(), predecir_df(beta, tr)),
                m_te=metricas(te[Y].to_numpy(), predecir_df(beta, te)))


def graficar(M, v=None, p=None):
    """Si hay datos ingresados (v) y su predicción (p), las gráficas se adaptan a ellos."""
    b = M["beta"]; te = M["te"]; real, pred = te[Y].to_numpy(), predecir_df(b, te)
    fig = Figure(figsize=(11, 4.6)); a1, a2 = fig.subplots(1, 2)
    a1.scatter(real, pred, c="#1f4e8c", alpha=.6, label="Datos de prueba"); a1.plot([0, 100], [0, 100], "r--", label="Predicción perfecta")
    if p is not None:
        a1.axhline(p, color="#e67e22", lw=1.5, ls=":"); a1.scatter([p], [p], s=140, marker="*", c="#e67e22", zorder=5, label=f"Su predicción: {p:.1f} %")
    a1.set(title=f"Real vs. predicho (prueba, R²={M['m_te'][2]:.2f})", xlabel="Cumplimiento real (%)", ylabel="Cumplimiento predicho (%)", xlim=(0, 100), ylim=(0, 100)); a1.legend(fontsize=8)
    etq = [f"X{i + 1}" for i in range(5)]
    if v is None:
        a2.barh(etq, b[1:], color=["#2e8b57" if c > 0 else "#c0392b" for c in b[1:]])
        a2.set(title="Efecto de cada variable (coeficientes)", xlabel="Puntos de cumplimiento por unidad")
    else:
        ap = b[1:] * np.array(v)                               # aporte = coeficiente x valor ingresado
        a2.barh(etq, ap, color=["#2e8b57" if c > 0 else "#c0392b" for c in ap])
        for i, c in enumerate(ap): a2.text(c, i, f" {c:+.1f} ", va="center", ha="left" if c >= 0 else "right", fontsize=8)
        a2.set(title=f"Aporte de sus datos (β·X)  |  base β0 = {b[0]:.1f}", xlabel="Puntos de cumplimiento aportados"); a2.margins(x=.2)
    a2.axvline(0, c="k", lw=.8); a2.invert_yaxis()
    fig.tight_layout()
    return fig


def texto_modelo(M):
    b = M["beta"]; nombres = ["Intercepto β0"] + ETIQ
    te = M["te"]; real, p = te[Y].to_numpy(), predecir_df(b, te)
    t = "ECUACIÓN NORMAL:  β = (XᵀX)⁻¹ · Xᵀ · Y\n\n"
    for nom, m in [("X  (matriz de datos)", M["X"]), ("Xᵀ (transpuesta)", M["X"].T), ("Y  (cumplimiento real)", M["y"]), ("β  (coeficientes)", b.reshape(-1, 1))]:
        t += f"{nom:<24}: {m.shape[0]} filas x {m.shape[1]} col.\n"
    t += f"\nPrimeras 5 filas de X (1, X1..X5):\n{np.round(M['X'][:5], 1)}\n\nCOEFICIENTES E INTERPRETACIÓN\n"
    for n, v in zip(nombres, b):
        t += f"  {n:<40} {v:+8.3f}"
        t += "  (valor base)\n" if n[0] == "I" else f"  -> +1 unidad {'SUBE' if v > 0 else 'BAJA'} el cumplimiento {abs(v):.2f} puntos\n"
    t += "\nMODELO:  Y = " + f"{b[0]:.2f} " + " ".join(f"{v:+.2f}·X{i + 1}" for i, v in enumerate(b[1:]))
    t += "\n\nEVALUACIÓN               MAE      RMSE     R²\n"
    t += f"  Entrenamiento (80%)  {M['m_tr'][0]:7.2f}  {M['m_tr'][1]:7.2f}  {M['m_tr'][2]:6.3f}\n"
    t += f"  Prueba (20%)         {M['m_te'][0]:7.2f}  {M['m_te'][1]:7.2f}  {M['m_te'][2]:6.3f}\n"
    t += "\nREAL -> PREDICHO (primeros 8 de prueba):  " + "  ".join(f"{r:.0f}->{q:.0f}" for r, q in zip(real[:8], p[:8]))
    return t


# ----------------------------------------------------------------- Interfaz
st.set_page_config(page_title="Predicción del cumplimiento académico", layout="wide")
st.title("Predicción del cumplimiento académico")
st.caption("Regresión lineal múltiple resuelta con la ecuación normal  β = (XᵀX)⁻¹ Xᵀ Y")

with st.sidebar:
    st.header("Datos")
    archivo = st.file_uploader("Cargar CSV (opcional)", type="csv")
    st.caption("Si no sube nada se usa `datos_estudiantes.csv`.")

try:
    df = pd.read_csv(archivo) if archivo else pd.read_csv(CSV_POR_DEFECTO)
    df[VARS + [Y]].astype(float)
except Exception:
    st.error(f"El CSV debe tener las columnas: {VARS + [Y]}")
    st.stop()

M = entrenar(df)
st.success(f"{len(df)} registros cargados. Modelo entrenado. R² en prueba = {M['m_te'][2]:.3f}")

t1, t2, t3, t4 = st.tabs(["1. Datos", "2. Modelo y coeficientes", "3. Predicción", "4. Gráficas"])

with t1:
    st.write(f"Registros: **{len(df)}**  (se muestran los primeros 50)")
    st.dataframe(df.head(50), width="stretch")
    c1, c2 = st.columns([3, 1])
    c1.subheader("Estadísticas"); c1.dataframe(df[VARS + [Y]].describe().round(2), width="stretch")
    c2.subheader("Correlación con el cumplimiento"); c2.dataframe(df[VARS + [Y]].corr()[Y].round(2), width="stretch")

with t2:
    st.code(texto_modelo(M), language=None)

with t3:
    st.subheader("Ingrese los datos del estudiante")
    col1, col2 = st.columns(2)
    with col1:
        v1 = st.number_input(ETIQ[0], min_value=0.0, value=12.0, step=1.0)
        v2 = st.number_input(ETIQ[1], min_value=0.0, value=8.0, step=1.0)
        v3 = st.number_input(ETIQ[2], min_value=1.0, max_value=5.0, value=4.0, step=1.0)
    with col2:
        v4 = st.number_input(ETIQ[3], min_value=0.0, value=15.0, step=1.0)
        v5 = st.number_input(ETIQ[4], min_value=0.0, value=2.0, step=1.0)
    if st.button("Predecir cumplimiento", type="primary"):
        st.session_state["v"] = [v1, v2, v3, v4, v5]
    if "v" in st.session_state:
        v = st.session_state["v"]
        p = float(predecir_df(M["beta"], pd.DataFrame([v], columns=VARS))[0])
        st.session_state["p"] = p
        st.metric("Cumplimiento estimado", f"{p:.1f} %")
        st.progress(min(max(p / 100, 0.0), 1.0))
        if p >= 80: st.success("Nivel ALTO: buen panorama.")
        elif p >= 60: st.warning("Nivel MEDIO: conviene organizar mejor el tiempo.")
        else: st.error("Nivel BAJO: riesgo de no entregar a tiempo; revise planificación y distracciones.")
        st.caption("Es una estimación estadística, no una garantía.")

with t4:
    v, p = st.session_state.get("v"), st.session_state.get("p")
    st.pyplot(graficar(M, v, p) if v is not None else graficar(M))
    st.caption("Las gráficas se adaptan a los datos ingresados en la pestaña 3." if v is None else "Gráficas actualizadas con los datos ingresados.")
