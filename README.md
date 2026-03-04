FitLife Streamlit Dashboard

Instrucciones rápidas

1. Instala dependencias:

```powershell
python -m pip install -r requirements.txt
```

(Si `python` no está en el PATH usa `py -m pip`.)

2. Ejecuta la app:

```powershell
streamlit run streamlit_app.py
```

3. La app mostrará las tablas `fitlife_context.csv` y `fitlife_members.csv` y varias visualizaciones.

Notas:
- Para generar informes adicionales puedes instalar `ydata-profiling`.
- Para exploración de tablas en ventana gráfica puedes instalar `pandasgui`.

## Analítica LLM-driven

El módulo `python.py` expone cuatro funciones auxiliares para un flujo
analítico:

* `generate_sql(question, csv_path=None, schema=None, semantics=None,
  golden_dataset=None, llm_client=None)` – convierte una pregunta en SQL
  usando un LLM (ej. OpenAI). Acepta distintos niveles de contexto
  (CSV, esquema, semánticas, dataset canónico) para mejorar la precisión.
* `run_query(sql, conn=None)` – ejecuta SQL en una base de datos sqlite y
  devuelve un `DataFrame`.
* `explain_results(df, question, llm_client)` – pide al LLM una explicación
  amigable para stakeholders basándose en el resultado de la consulta.
* `add_visualization(df, output_path=None)` – genera un gráfico simple
  (histograma, barra o línea según los tipos) siguiendo guías del Wall
  Street Journal.

Puedes ver `main.py` como ejemplo de cómo encadenar estas funciones.

## Pruebas

Se incluye un conjunto básico de pruebas en `tests/test_sql_helpers.py`.
Ejecuta:

```powershell
pytest -q
```

debes tener `matplotlib` y `pytest` instalados (ya están en
`requirements.txt`).
