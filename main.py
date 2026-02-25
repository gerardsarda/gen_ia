import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# imports opcionales
try:
    from pandasgui import show
except ImportError:  # pragma: no cover
    show = None

try:
    from ydata_profiling import ProfileReport
except ImportError:  # pragma: no cover
    ProfileReport = None


def explore(df: pd.DataFrame, name: str) -> None:
    """Imprime un resumen básico de df y genera algunos gráficos."""
    print(f"\n\n===== Exploración de {name} =====")
    print("\nPrimeras filas:")
    print(df.head())
    print("\nInformación general:")
    df.info()
    print("\nEstadísticas descriptivas:")
    print(df.describe(include="all"))
    print("\nValores nulos por columna:")
    print(df.isna().sum())
    for col in df.select_dtypes(include="object").columns:
        print(f"\nConteo de valores para '{col}':")
        print(df[col].value_counts(dropna=False).head(20))
    num = df.select_dtypes(include="number")
    if not num.empty:
        num.hist(bins=15, figsize=(10, 6))
        plt.suptitle(f"Histogramas de {name}")
        plt.tight_layout()
        plt.show()
    if len(num.columns) > 1:
        sns.pairplot(num)
        plt.suptitle(f"Pairplot de {name}", y=1.02)
        plt.show()


def make_profile(df: pd.DataFrame, name: str) -> None:
    if ProfileReport is None:
        return
    print(f"Generando informe rápido para {name}...")
    rep = ProfileReport(df, title=f"Reporte {name}", minimal=True)
    file_name = f"{name}_report.html"
    rep.to_file(file_name)
    print(f"Informe guardado en {file_name}")


def main():
    print("Cargando datos...")
    context = pd.read_csv("fitlife_context.csv")
    members = pd.read_csv("fitlife_members.csv")
    explore(context, "fitlife_context")
    explore(members, "fitlife_members")
    if ProfileReport is not None:
        make_profile(context, "fitlife_context")
        make_profile(members, "fitlife_members")
    else:
        print("\nPara generar informes interactivos instala ydata-profiling:")
        print("    python -m pip install ydata-profiling")
    if show is not None:
        print("\nAbriendo visor interactivo de pandasgui...")
        show(context, members)
    else:
        print("\nPara ver los datos en una interface gráfica instala pandasgui:")
        print("    python -m pip install pandasgui")
        print("También puedes abrir los CSV en Excel u otro editor de hojas de cálculo.")


if __name__ == "__main__":
    main()
