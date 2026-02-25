import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FitLife Dashboard", page_icon="🔥", layout="wide")

# ── COLORES ──────────────────────────────────────────────────────────────────
BG      = "#0d0d0d"
SURFACE = "#161616"
BORDER  = "#2a2a2a"
ORANGE  = "#ff5500"
BLUE    = "#00b4d8"
GREEN   = "#00e676"
YELLOW  = "#ffd100"
RED     = "#ff3b3b"
MUTED   = "#555555"
TEXT    = "#f0f0f0"
PALETTE = [ORANGE, BLUE, GREEN, YELLOW, RED, "#a855f7", "#ec4899"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background:{BG}; }}
[data-testid="stSidebar"] {{ background:{SURFACE} !important; border-right:1px solid {BORDER} !important; }}
[data-testid="stMetric"] {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px;
    padding:18px 20px !important; border-top:2px solid {ORANGE};
}}
[data-testid="stMetricLabel"] p {{ font-size:10px !important; text-transform:uppercase; letter-spacing:2px; color:{MUTED} !important; }}
[data-testid="stMetricValue"] {{ font-family:'Bebas Neue',sans-serif !important; font-size:44px !important; color:{TEXT} !important; line-height:1 !important; }}
.stTabs [data-baseweb="tab-list"] {{ background:transparent; gap:4px; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    font-size:11px; text-transform:uppercase; letter-spacing:1.5px;
    border-radius:8px !important; border:none !important;
}}
.stTabs [aria-selected="true"] {{ background:rgba(255,85,0,0.15) !important; color:{ORANGE} !important; }}
.stDownloadButton button {{
    background:rgba(255,85,0,0.12) !important; color:{ORANGE} !important;
    border:1px solid rgba(255,85,0,0.3) !important; border-radius:10px !important;
}}
hr {{ border-color:{BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB DARK ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.6,
    "legend.facecolor": SURFACE, "legend.edgecolor": BORDER,
})


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ctx = pd.read_csv("fitlife_context.csv")
    mem = pd.read_csv("fitlife_members.csv")
    return ctx, mem

context, members = load_data()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='font-family:Bebas Neue,sans-serif;font-size:30px;"
        f"color:{ORANGE};letter-spacing:4px;padding:8px 0 4px'>FITLIFE</div>"
        f"<div style='font-size:10px;color:{MUTED};letter-spacing:2px;margin-bottom:20px'>DASHBOARD 2025</div>",
        unsafe_allow_html=True
    )
    st.divider()

    dataset  = st.selectbox("Dataset", ["members", "context"])
    show_n   = st.slider("Filas a mostrar", 5, 100, 15)
    df       = context if dataset == "context" else members
    num_cols = df.select_dtypes(include="number").columns.tolist()

    plot_cols = []
    if num_cols:
        plot_cols = st.multiselect("Columnas a graficar", num_cols, default=num_cols[:4])

    st.divider()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Exportar CSV", data=csv,
                       file_name=f"{dataset}.csv", mime="text/csv")
    st.markdown(
        f"<div style='font-size:10px;color:{MUTED};margin-top:12px'>"
        f"{df.shape[0]:,} filas · {df.shape[1]} columnas</div>",
        unsafe_allow_html=True
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-family:Bebas Neue,sans-serif;font-size:50px;"
    f"letter-spacing:3px;line-height:1;margin-bottom:4px'>"
    f"{dataset.upper()} <span style='color:{ORANGE}'>·</span> VISTA GENERAL</h1>"
    f"<p style='color:{MUTED};font-family:JetBrains Mono,monospace;"
    f"font-size:11px;letter-spacing:1px;margin-bottom:24px'>"
    f"{df.shape[0]:,} FILAS · {df.shape[1]} COLUMNAS · "
    f"{df.isna().sum().sum()} NULOS TOTALES</p>",
    unsafe_allow_html=True
)

# ── KPI METRICS ───────────────────────────────────────────────────────────────
num = df.select_dtypes(include="number")
if not num.empty:
    kpi_cols = st.columns(min(4, len(num.columns)))
    for i, col in enumerate(num.columns[:4]):
        with kpi_cols[i]:
            st.metric(
                label=col.upper().replace("_", " "),
                value=f"{df[col].sum():,.0f}",
                delta=f"media {df[col].mean():,.1f}"
            )

st.divider()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_table, tab_stats, tab_charts, tab_corr, tab_nulls = st.tabs([
    "📋  Tabla", "📐  Estadísticas", "📊  Gráficos", "🔗  Correlación", "🕳  Nulos"
])


# ── TAB 1: TABLA ─────────────────────────────────────────────────────────────
with tab_table:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("MUESTRA DE DATOS")
        st.dataframe(df.head(show_n), use_container_width=True, height=420)
    with c2:
        st.caption("TIPOS Y NULOS POR COLUMNA")
        dtype_df = pd.DataFrame({
            "tipo":   df.dtypes.astype(str),
            "nulos":  df.isna().sum(),
            "nulos%": (df.isna().mean() * 100).round(1)
        })
        st.dataframe(dtype_df, use_container_width=True, height=420)


# ── TAB 2: ESTADÍSTICAS ───────────────────────────────────────────────────────
with tab_stats:
    st.caption("ESTADÍSTICAS DESCRIPTIVAS")
    desc = df.describe(include="all").T
    desc.insert(0, "dtype", df.dtypes.astype(str))
    st.dataframe(desc, use_container_width=True, height=480)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        st.divider()
        st.caption("TOP VALORES — COLUMNAS CATEGÓRICAS")
        cat_sel = st.selectbox("Columna", cat_cols)
        vc = df[cat_sel].value_counts().head(15).reset_index()
        vc.columns = [cat_sel, "count"]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.barh(vc[cat_sel].astype(str)[::-1], vc["count"][::-1],
                       color=ORANGE, alpha=0.85, height=0.6)
        ax.bar_label(bars, padding=5, color=MUTED, fontsize=8,
                     fmt=lambda x: f"{int(x):,}")
        ax.set_xlabel("Frecuencia")
        ax.set_title(f"Top 15 — {cat_sel}", pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── TAB 3: GRÁFICOS ───────────────────────────────────────────────────────────
with tab_charts:
    if not plot_cols:
        st.info("Selecciona columnas numéricas en el panel lateral.")
    else:
        # Histogramas + KDE
        st.caption("DISTRIBUCIÓN — HISTOGRAMAS")
        ncols = 2
        nrows = (len(plot_cols) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, 4 * nrows),
                                 constrained_layout=True)
        axes = np.array(axes).flatten()

        for i, col in enumerate(plot_cols):
            ax    = axes[i]
            color = PALETTE[i % len(PALETTE)]
            data  = df[col].dropna()

            sns.histplot(data, ax=ax, kde=True, color=color, alpha=0.7,
                         linewidth=0,
                         line_kws={"linewidth": 2, "color": "white"})
            ax.axvline(data.mean(), color="white", linewidth=1,
                       linestyle="--", alpha=0.6)
            ax.text(data.mean(), ax.get_ylim()[1] * 0.95,
                    f" μ={data.mean():,.1f}", color="white",
                    fontsize=8, va="top")
            ax.set_title(col.replace("_", " ").upper())
            ax.set_xlabel("")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        st.pyplot(fig)
        plt.close()

        # Boxplots
        st.caption("DISTRIBUCIÓN — BOXPLOTS")
        fig2, ax2 = plt.subplots(figsize=(14, 3.5))
        data_box = [df[c].dropna().values for c in plot_cols]
        bp = ax2.boxplot(data_box, patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2},
                         whiskerprops={"color": MUTED},
                         capprops={"color": MUTED},
                         flierprops={"marker": "o", "markersize": 3,
                                     "markerfacecolor": ORANGE, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax2.set_xticks(range(1, len(plot_cols) + 1))
        ax2.set_xticklabels([c.replace("_", " ") for c in plot_cols], fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ── TAB 4: CORRELACIÓN ────────────────────────────────────────────────────────
with tab_corr:
    if len(num.columns) < 2:
        st.info("Se necesitan al menos 2 columnas numéricas.")
    else:
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.9),
                                        max(5, len(corr) * 0.8)))
        sns.heatmap(
            corr, ax=ax, annot=True, fmt=".2f", linewidths=0.5,
            linecolor=BG,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            vmin=-1, vmax=1, square=True,
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.7}
        )
        ax.set_title("Matriz de correlación de Pearson", pad=16)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top pares
        st.divider()
        st.caption("PARES MÁS CORRELACIONADOS")
        pairs = (
            corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
            .stack().reset_index()
        )
        pairs.columns = ["col_a", "col_b", "correlación"]
        pairs = pairs.reindex(pairs["correlación"].abs()
                              .sort_values(ascending=False).index)
        st.dataframe(pairs.head(15), use_container_width=True)


# ── TAB 5: NULOS ──────────────────────────────────────────────────────────────
with tab_nulls:
    null_s = df.isna().sum()
    null_s = null_s[null_s > 0].sort_values(ascending=False)

    if null_s.empty:
        st.success("✅  No hay valores nulos en este dataset.")
    else:
        pct = (null_s / len(df) * 100).round(1)
        c_a, c_b = st.columns([2, 1])

        with c_a:
            fig, ax = plt.subplots(figsize=(10, max(3, len(null_s) * 0.55)))
            colors = [RED if p > 20 else YELLOW if p > 5 else ORANGE
                      for p in pct.values[::-1]]
            bars = ax.barh(null_s.index[::-1], pct.values[::-1],
                           color=colors, height=0.55)
            ax.bar_label(bars, labels=[f"{v}%" for v in pct.values[::-1]],
                         padding=5, color=MUTED, fontsize=8)
            ax.set_xlabel("% de nulos")
            ax.set_title("Valores nulos por columna", pad=12)
            ax.set_xlim(0, min(100, pct.max() * 1.3))
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(left=False)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c_b:
            st.caption("DETALLE")
            detail = pd.DataFrame({"nulos": null_s, "% total": pct})
            st.dataframe(detail, use_container_width=True, height=400)
