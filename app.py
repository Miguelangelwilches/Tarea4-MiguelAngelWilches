import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Terremotos Significativos", page_icon="🌍", layout="wide")
st.title("🌍 Análisis de Terremotos Significativos")
st.subheader("Datos registrados entre 1965 y 2016")
st.divider()

# CARGA DE DATOS — Método simplificado
@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    try:
        # Intento 1: archivo local (más común)
        df = pd.read_csv("dataset.csv")
    except:
        try:
            # Intento 2: con ruta relativa
            df = pd.read_csv("./dataset.csv")
        except Exception as e:
            st.error(f"No se pudo cargar el archivo de datos. Error: {type(e).__name__}")
            st.warning("Verifica que el archivo dataset.csv esté subido en tu repositorio de GitHub.")
            st.stop()
    
    # Procesamiento de datos
    bins = [5.5, 5.8, 6.2, 6.7, np.inf]
    etiquetas = ['5.5 - 5.7 (Leve)', 
                 '5.8 - 6.1 (Moderado)',
                 '6.2 - 6.6 (Fuerte)', 
                 '≥ 6.7 (Muy fuerte)']
    df['Categoria_Magnitud'] = pd.cut(df['Magnitud'], bins=bins, labels=etiquetas, include_lowest=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Anio'] = df['Date'].dt.year
    
    return df

# Ejecutar carga
df = cargar_datos()

# Si llegamos aquí, los datos cargaron bien
st.success(f"✅ Datos cargados correctamente: {len(df):,} registros")
st.subheader("Vista previa de los datos")
st.dataframe(df.head(), use_container_width=True)

# --- Aquí van tus gráficos y análisis ---
st.divider()
st.subheader("📊 Resumen del conjunto de datos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de registros", f"{len(df):,}")
with col2:
    st.metric("Magnitud promedio", f"{df['Magnitud'].mean():.2f}")
with col3:
    st.metric("Año más reciente", f"{df['Anio'].max():.0f}")

st.subheader("Distribución por categoría de magnitud")
conteo = df['Categoria_Magnitud'].value_counts()
st.bar_chart(conteo)


st.subheader("Distribución por categoría de magnitud")
conteo = df['Categoria_Magnitud'].value_counts()
st.bar_chart(conteo)

