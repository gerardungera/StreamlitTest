import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# configuración de la página
st.set_page_config(page_title="Julia Set Explorer", layout="wide")

# ejemplo de fractales de la clase 5
@st.cache_data(show_spinner="Calculando fractal...")
def calculate_julia(width, height, max_iterations, c_real, c_imag, x_range, y_range):
    """
    Cálculo optimizado usando NumPy para evitar bucles anidados lentos.
    """
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    C = complex(c_real, c_imag)
    
    output = np.zeros(Z.shape, dtype=int)
    mask = np.full(Z.shape, True, dtype=bool)
    
    for i in range(max_iterations):
        Z[mask] = Z[mask] * Z[mask] + C
        diverged = np.abs(Z) > 2
        escaping_now = diverged & mask
        output[escaping_now] = i
        mask[diverged] = False
        
    return output

st.title("Explorador del Conjunto de Julia")

st.markdown("""
Esta aplicación demuestra el uso de **Streamlit** para visualizar fractales complejos, 
aplicando técnicas de **caching** y **paralelismo de datos** con NumPy.
""")

# sidebar para parámetros
st.sidebar.header("Configuración del Fractal")
c_r = st.sidebar.slider("C Real", -2.0, 2.0, -0.62772, step=0.01)
c_i = st.sidebar.slider("C Imaginario", -2.0, 2.0, -0.42193, step=0.01)

st.sidebar.divider()
res = st.sidebar.select_slider("Resolución (Ancho/Alto)", options=[250, 500, 750, 1000], value=500)
iters = st.sidebar.number_input("Máx. Iteraciones", 50, 1000, 300)

# limpiar caché si es necesario
if st.sidebar.button("Limpiar Caché"):
    st.cache_data.clear()
    st.rerun()

col1, col2 = st.columns([3, 1])

with col1:
    start_time = time.time()
    julia_data = calculate_julia(res, res, iters, c_r, c_i, (-1.8, 1.8), (-1.8, 1.8))
    end_time = time.time()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(julia_data, extent=[-1.8, 1.8, -1.8, 1.8], cmap='magma')
    plt.axis('off')
    st.pyplot(fig)

with col2:
    st.metric("Tiempo de Cálculo", f"{end_time - start_time:.4f} s")
    st.info(f"El resultado para C = {c_r} + {c_i}j ha sido {'recuperado del caché' if (end_time - start_time) < 0.01 else 'calculado en tiempo real'}.")
    
    st.subheader("¿Qué estamos viendo?")
    st.write("""
    El Conjunto de Julia se define por la iteración de la función:
    """)
    st.latex(r"z_{n+1} = z_n^2 + c")
    st.write("Donde $c$ es un número complejo constante que tú controlas desde el panel lateral.")
