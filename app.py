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

# CARGA DE DATOS
@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    try:
        df = pd.read_csv("dataset.csv")
    except:
        try:
            df = pd.read_csv("./dataset.csv")
        except Exception as e:
            st.error(f"No se pudo cargar el archivo de datos. Error: {type(e).__name__}")
            st.warning("Verifica que el archivo dataset.csv esté subido en tu repositorio de GitHub.")
            st.stop()
    
    # Normalizar nombres
    df.columns = df.columns.str.strip()
    
    # Buscar columna de magnitud
    col_magnitud = None
    for col in df.columns:
        if 'magnitud' in col.lower():
            col_magnitud = col
            break
    
    # Crear categorías de magnitud
    bins = [5.5, 5.8, 6.2, 6.7, np.inf]
    etiquetas = ['5.5 - 5.7 (Leve)', 
                 '5.8 - 6.1 (Moderado)',
                 '6.2 - 6.6 (Fuerte)', 
                 '≥ 6.7 (Muy fuerte)']
    df['Categoria_Magnitud'] = pd.cut(df[col_magnitud], bins=bins, labels=etiquetas, include_lowest=True)
    
    # Buscar columna de fecha
    col_fecha = None
    for col in df.columns:
        if 'date' in col.lower() or 'fecha' in col.lower():
            col_fecha = col
            break
    
    if col_fecha:
        df['Date'] = pd.to_datetime(df[col_fecha], errors='coerce')
        df['Anio'] = df['Date'].dt.year
    
    return df, col_magnitud

# Ejecutar carga
df, col_magnitud = cargar_datos()

# --- RESUMEN GENERAL ---
st.success(f"✅ Datos cargados correctamente: {len(df):,} registros | Periodo: {df['Anio'].min():.0f} - {df['Anio'].max():.0f}")

with st.expander("📋 Ver datos completos"):
    st.dataframe(df, use_container_width=True)

st.divider()
st.header(" 1. Resumen del Conjunto de Datos")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de registros", f"{len(df):,}")
with col2:
    st.metric("Magnitud promedio", f"{df[col_magnitud].mean():.2f}")
with col3:
    st.metric("Magnitud máxima", f"{df[col_magnitud].max():.1f}")
with col4:
    st.metric("Magnitud mínima", f"{df[col_magnitud].min():.1f}")

st.info("""
** Interpretación:** El conjunto de datos cuenta con 23,412 registros de sismos significativos 
registrados durante 52 años. La magnitud promedio es de aproximadamente 5.88, con valores 
que van desde 5.5 hasta más de 8.0 en la escala de Richter.
""")

st.divider()
st.header(" 2. Distribución por Categoría de Magnitud")

conteo = df['Categoria_Magnitud'].value_counts()
st.bar_chart(conteo, use_container_width=True)

st.info("""
** Interpretación:** La mayoría de los sismos registrados corresponden a la categoría 
"Leve" (5.5 - 5.7), seguidos por los "Moderados". A medida que aumenta la magnitud, 
la frecuencia disminuye, siendo los sismos "Muy fuertes" los menos comunes pero los 
que tienen mayor impacto.
""")

st.divider()
st.header(" 3. Evolución de Sismos por Año")

sismos_por_anio = df.groupby('Anio').size()
st.line_chart(sismos_por_anio, use_container_width=True)

st.info("""
** Interpretación:** El gráfico muestra la cantidad de sismos registrados cada año. 
Se observa que la actividad sísmica ha variado a lo largo de las décadas, con algunos 
periodos de mayor actividad que otros. Esto puede deberse tanto a cambios reales en 
la actividad sísmica como a mejoras en la tecnología de detección a lo largo del tiempo.
""")

st.divider()
st.header(" 4. Distribución de la Magnitud")

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df[col_magnitud], bins=30, kde=True, color='#2ecc71', ax=ax)
ax.set_xlabel('Magnitud')
ax.set_ylabel('Cantidad de sismos')
ax.set_title('Histograma de Magnitudes')
st.pyplot(fig)

st.info("""
** Interpretación:** La distribución de magnitudes se concentra principalmente entre 
5.5 y 6.5, formando una distribución asimétrica hacia la derecha. Esto confirma que 
los sismos de gran magnitud son eventos menos frecuentes.
""")

st.divider()
st.header(" 5. Profundidad vs. Magnitud")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Magnitude', y='Depth', data=df, alpha=0.5, color='#3498db', ax=ax2)
ax2.set_xlabel('Magnitud')
ax2.set_ylabel('Profundidad (km)')
ax2.set_title('Relación entre Profundidad y Magnitud')
st.pyplot(fig2)

st.info("""
** Interpretación:** En este gráfico se observa la relación entre la profundidad 
del sismo y su magnitud. No se aprecia una correlación fuerte, lo que indica que un 
sismo puede ser fuerte tanto en la superficie como a gran profundidad.
""")

# --- SECCIÓN DE INTELIGENCIA ARTIFICIAL ---
st.divider()
st.header(" Uso de Inteligencia Artificial en el Análisis")

st.subheader(" Bitácora de Actividades con IA")

st.markdown("""
| Actividad | Descripción |
|---|---|
| **Consulta inicial** | Solicitud de guía paso a paso para importar y analizar un conjunto de datos sísmicos en Google Colab |
| **Carga de datos** | La IA proporcionó el código para importar el archivo CSV desde Kaggle y desde enlace directo en GitHub |
| **Análisis descriptivo** | Se pidió apoyo para calcular e interpretar métricas como media, mediana, desviación estándar y coeficiente de variación |
| **Matriz de correlación** | La IA ayudó a interpretar los valores de correlación entre variables, explicando la diferencia entre correlación y causalidad |
| **Generación de gráficos** | Se solicitó código para visualizar distribuciones, tendencias temporales y relaciones entre variables |
| **Resolución de errores** | La IA ayudó a identificar y corregir errores de nombres de columnas, rutas de archivos y formato de fechas |
| **Construcción del Dashboard** | Apoyo en la estructura, diseño y desplazamiento de la aplicación en Streamlit Cloud |
""")

st.subheader(" Ejemplo de Consulta Realizada")

with st.expander("Ver prompt enviado a la IA"):
    st.code("""
Actualmente estoy trabajando con un conjunto de datos de terremotos significativos 
registrados entre 1965 y 2016. Necesito que me ayudes a realizar lo siguiente:

1. Importar los datos en Google Colab
2. Realizar un análisis descriptivo de variables numéricas y categóricas
3. Calcular e interpretar la matriz de correlación
4. Crear visualizaciones que muestren:
   - Distribución por categoría de magnitud
   - Evolución temporal de los sismos
   - Relación entre profundidad y magnitud

Por favor, incluye el código en Python y una interpretación clara de cada resultado.
    """, language="text")

st.subheader(" Comparación: Análisis Propio vs. Resultados de la IA")

col_ia1, col_ia2 = st.columns(2)

with col_ia1:
    st.markdown("###  Mi Análisis")
    st.markdown("""
    - Identifiqué que la mayoría de sismos están entre 5.5 y 6.1 de magnitud
    - Observé que hay más sismos registrados en años recientes
    - Noté que no hay relación clara entre profundidad y magnitud
    - Organicé los datos y las categorías de magnitud
    """)

with col_ia2:
    st.markdown("###  Aportes de la IA")
    st.markdown("""
    - Explicó por qué podría aumentar el número de sismos registrados (mejoras en tecnología)
    - Interpretó el significado del coeficiente de variación
    - Aclaró que correlación NO implica causalidad
    - Sugirió mejores prácticas de visualización y limpieza de datos
    - Resolvió errores de ejecución que no lograba identificar solo
    """)

st.subheader(" Evaluación Crítica")

st.success("""
** Aciertos y utilidad de la IA:**
- La IA aceleró el aprendizaje al explicar conceptos que costaba entender
- Ayudó a identificar errores en el código que habrían tomado mucho tiempo solucionar
- Ofreció diferentes formas de visualizar los datos, permitiendo elegir la más clara
- Las interpretaciones sirvieron como guía pero siempre fueron verificadas con los propios datos

** Limitaciones observadas:**
- En ocasiones la IA sugería rutas de archivo incorrectas que no funcionaban en el entorno de Streamlit
- Algunos códigos debían adaptarse porque suponía nombres de columnas distintos a los reales
- Siempre fue necesario verificar que los resultados tuvieran sentido lógico con el contexto del problema

** Reflexión final:**
> La Inteligencia Artificial es una herramienta poderosa, pero **no reemplaza el pensamiento crítico**. 
> Es responsabilidad de quien la usa verificar, validar y entender cada resultado. En este trabajo, 
> la IA funcionó como un tutor disponible en cualquier momento, guiando el proceso sin dar 
> las respuestas finales sin antes verificar que se comprendiera lo que se estaba haciendo.
""")

st.divider()
st.header(" Conclusiones Generales")


st.write("""
A partir del análisis realizado con los datos de sismos significativos entre 1965 y 2016, 
se pueden destacar las siguientes conclusiones:

- **🔹 Predominan los sismos de magnitud moderada:** Más de la mitad de los eventos registrados 
  tienen una magnitud entre 5.5 y 6.1, siendo los eventos extremos los menos frecuentes.

- **🔹 Estabilidad en la actividad:** La cantidad anual de sismos ha mostrado cierta variación, 
  sin embargo, esto puede estar influenciado por una mejor cobertura de estaciones de monitoreo.

- **🔹 No hay relación directa entre profundidad y magnitud:** Los sismos pueden ocurrir a 
  distintas profundidades independientemente de su intensidad.

- **🔹 Importancia del monitoreo continuo:** Conocer el comportamiento histórico de los sismos 
  ayuda a entender el riesgo sísmico y prepararse mejor ante eventos futuros.

>  *El análisis de datos sísmicos es fundamental para la prevención de desastres naturales 
> y la seguridad de las comunidades en zonas de riesgo.*
""")

st.divider()
st.caption(" Análisis realizado con datos de terremotos significativos | Dashboard creado para el trabajo de análisis de datos")

st.subheader("Distribución por categoría de magnitud")
conteo = df['Categoria_Magnitud'].value_counts()
st.bar_chart(conteo)

