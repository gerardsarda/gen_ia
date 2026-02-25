import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FitLife Dashboard", layout="wide")
st.title("FitLife — Dashboard rápido")

@st.cache_data
def load_data():
    ctx = pd.read_csv("fitlife_context.csv")
    mem = pd.read_csv("fitlife_members.csv")
    return ctx, mem

context, members = load_data()

# Sidebar
st.sidebar.header("Controles")
dataset = st.sidebar.selectbox("Dataset", ["context", "members"])
show_n = st.sidebar.slider("Filas a mostrar", 5, 50, 10)

df = context if dataset == "context" else members

# Summary
st.header(f"Vista rápida — {dataset}")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Tabla (muestra)")
    st.dataframe(df.head(show_n), use_container_width=True)
    st.markdown(f"**Filas:** {df.shape[0]}  \n**Columnas:** {df.shape[1]}")

with col2:
    st.subheader("Tipos y nulos")
    dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
    dtypes["n_null"] = df.isna().sum().values
    st.dataframe(dtypes, use_container_width=True)

# Descriptive stats
st.subheader("Estadísticas descriptivas")
st.dataframe(df.describe(include='all').T, use_container_width=True)

# Gráficos
st.subheader("Gráficos — variables numéricas")
num = df.select_dtypes(include="number")
if num.shape[1] == 0:
    st.info("No hay columnas numéricas en este dataset.")
else:
    cols = st.multiselect("Columnas para graficar", list(num.columns), default=list(num.columns)[:3])
    if cols:
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 3*len(cols)), constrained_layout=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, cols):
            sns.histplot(df[c].dropna(), ax=ax, kde=True, color="#2b8cbe")
            ax.set_title(c)
        st.pyplot(fig)

    if num.shape[1] > 1:
        st.subheader("Mapa de correlación")
        corr = num.corr()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax2)
        st.pyplot(fig2)

# Conteos rápidos para members
if dataset == 'members':
    st.subheader('Conteos rápidos')
    for col in df.select_dtypes(include='object').columns[:3]:
        st.markdown(f"**Top valores — {col}**")
        st.table(df[col].value_counts().head(10))

# Download
st.sidebar.markdown("---")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Descargar CSV", data=csv, file_name=f"{dataset}.csv", mime='text/csv')

st.sidebar.markdown("---\nGenerado con Streamlit")
