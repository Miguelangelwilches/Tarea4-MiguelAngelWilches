# ==============================
# LIBRERÍAS NECESARIAS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(
    page_title="Terremotos Significativos - Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Análisis de Terremotos Significativos")
st.subheader("Datos registrados entre 1965 y 2016")
st.markdown("---")

# ==============================
# CARGA DE DATOS
# ==============================
@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset.csv")
    
    bins = [5.5, 5.8, 6.2, 6.7, np.inf]
    etiquetas = ['5.5 - 5.7 (Leve)', '5.8 - 6.1 (Moderado)', 
                 '6.2 - 6.6 (Fuerte)', '≥ 6.7 (Muy fuerte)']
    df['Categoria_Magnitud'] = pd.cut(df['Magnitude'], bins=bins, labels=etiquetas, include_lowest=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Anio'] = df['Date'].dt.year
    
    return df

df = cargar_datos()

# ==============================
# BARRA LATERAL: FILTROS
# ==============================
st.sidebar.header("🔍 Filtros de búsqueda")

categorias = ['Todas'] + list(df['Categoria_Magnitud'].dropna().unique())
cat_seleccionada = st.sidebar.selectbox("Categoría de magnitud", categorias)

min_prof = int(df['Depth'].min())
max_prof = int(df['Depth'].max())
rango_profundidad = st.sidebar.slider("Rango de profundidad (km)", 
                                        min_value=min_prof, 
                                        max_value=max_prof, 
                                        value=(min_prof, max_prof))

df_filtrado = df.copy()
if cat_seleccionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['Categoria_Magnitud'] == cat_seleccionada]

df_filtrado = df_filtrado[
    (df_filtrado['Depth'] >= rango_profundidad[0]) & 
    (df_filtrado['Depth'] <= rango_profundidad[1])
]

st.sidebar.markdown(f"✅ **{len(df_filtrado)} registros encontrados**")

# ==============================
# PESTAÑAS
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📊 Gráficos", "📐 Análisis Estadístico", "📖 Hallazgos"])

with tab1:
    st.header("📋 Exploración de los Datos")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de registros", value=f"{len(df_filtrado):,}")
    with col2:
        st.metric("Magnitud promedio", value=f"{df_filtrado['Magnitude'].mean():.2f}")
    with col3:
        st.metric("Profundidad promedio (km)", value=f"{df_filtrado['Depth'].mean():.1f}")
    st.subheader("Tabla de datos")
    st.dataframe(df_filtrado, use_container_width=True)

with tab2:
    st.header("📊 Visualizaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sismos por categoría")
        conteo = df_filtrado['Categoria_Magnitud'].value_counts()
        fig1, ax1 = plt.subplots()
        colores = sns.color_palette("Reds", len(conteo))
        ax1.pie(conteo.values, labels=conteo.index, autopct='%1.1f%%', colors=colores)
        ax1.axis('equal')
        st.pyplot(fig1)
    with col2:
        st.subheader("Profundidad vs Magnitud")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Depth', y='Magnitude', data=df_filtrado, alpha=0.6, ax=ax2, color='darkred')
        ax2.set_xlabel("Profundidad (km)")
        ax2.set_ylabel("Magnitud")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    st.subheader("Evolución en el tiempo")
    tendencia = df_filtrado.groupby('Anio')['Magnitude'].mean()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=tendencia.index, y=tendencia.values, marker='o', ax=ax3, color='darkred')
    ax3.set_xlabel("Año")
    ax3.set_ylabel("Magnitud promedio")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with tab3:
    st.header("📐 Análisis Estadístico")
    st.markdown("### Comparación de profundidad: Sismos leves vs. Sismos fuertes")
    
    grupo_leve = df[df['Categoria_Magnitud'].isin(['5.5 - 5.7 (Leve)', '5.8 - 6.1 (Moderado)'])]['Depth'].dropna()
    grupo_fuerte = df[df['Categoria_Magnitud'].isin(['6.2 - 6.6 (Fuerte)', '≥ 6.7 (Muy fuerte)'])]['Depth'].dropna()
    
    t_stat, p_val = stats.ttest_ind(grupo_leve, grupo_fuerte, equal_var=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Profundidad promedio — Leves/Moderados: **{grupo_leve.mean():.1f} km**")
        st.info(f"📊 Profundidad promedio — Fuertes/Muy fuertes: **{grupo_fuerte.mean():.1f} km**")
    with col2:
        st.write(f"Valor de t: **{t_stat:.3f}**")
        st.write(f"Valor p: **{p_val:.6f}**")
    
    st.markdown("---")
    if p_val < 0.05:
        st.success("✅ **Conclusión:** Existe una diferencia estadísticamente significativa.")
    else:
        st.warning("⚠️ **Conclusión:** No hay evidencia suficiente para decir que son diferentes.")
    
    st.markdown("📝 En lenguaje sencillo: Comparamos la profundidad de sismos pequeños contra los grandes. La prueba nos dice si la diferencia es real o pudo ser casualidad.")

with tab4:
    st.header("📖 Lo que encontramos — En lenguaje claro")
    st.subheader("🔹 ¿Qué analicé?")
    st.write("Analizamos terremotos registrados entre 1965 y 2016, más de 2,300 eventos en total. Queríamos saber dónde ocurren, qué tan fuertes son y si han cambiado con el tiempo.")
    
    st.subheader("🔹 ¿Qué encontré?")
    st.write("- La mayoría de sismos son de magnitud leve a moderada. Solo 1 de cada 20 es muy fuerte.\n- Ocurren a profundidades muy distintas, desde muy cerca de la superficie hasta más de 600 km.\n- No se ha visto que sean más fuertes ahora que hace 50 años.\n- Donde hay más estaciones de monitoreo, se registran más detalles — no significa que haya más sismos.")
    
    st.subheader("🔹 ¿Qué significa esto?")
    st.write("Que los más fuertes sean menos frecuentes es buena noticia. Pero cuando ocurren cerca de la superficie, se sienten con mucha más fuerza. No hay señales de que la actividad esté aumentando, pero siempre debemos estar preparados.")
    
    st.subheader("🔹 ¿Qué se recomendaría?")
    st.write("- Conocer los sismos de tu región ayuda a construir casas más seguras.\n- Las zonas con sismos poco profundos necesitan mayor preparación.\n- Mantener las estaciones de monitoreo funcionando nos da mejor información.\n- No debemos bajar la guardia: un solo evento fuerte puede ser muy destructivo.")

st.markdown("---")
st.markdown("📌 **Elaborado por:** Miguel Ángel Wilches | UNAD | Análisis Exploratorio de Datos")
