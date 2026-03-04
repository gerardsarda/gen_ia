import sqlite3
import pandas as pd

# bring in the helpers we built in python.py
from python import generate_sql, run_query, explain_results, add_visualization

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def load_csv_into_sqlite(conn: sqlite3.Connection, csv_path: str, table_name: str):
    """Load a CSV file into a sqlite table (replacing if it exists)."""
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, index=False, if_exists="replace")


def main():
    # configure your LLM client however you want; environment variables
    # or other secret-handling is advised in production.
    if OpenAI is None:
        raise RuntimeError("openai library not installed, cannot run example")

    client = OpenAI(api_key="YOUR_API_KEY_HERE")

    # create an ephemeral database and populate it from our CSVs
    conn = sqlite3.connect(":memory:")
    load_csv_into_sqlite(conn, "fitlife_members.csv", "members")
    load_csv_into_sqlite(conn, "fitlife_context.csv", "context")

    question = input("Ask a question about the FitLife data: ")

    # generate SQL from the question; we pass a sample csv so the model
    # can infer schema if needed.
    sql = generate_sql(
        question,
        csv_path="fitlife_members.csv",
        llm_client=client,
    )
    print("\nGenerated SQL:\n", sql)

    df = run_query(sql, conn)
    print("\nExplanation for stakeholders:\n")
    print(explain_results(df, question, client))

    # make a quick chart saved to file; open or display as desired
    add_visualization(df, output_path="query_result.png")


if __name__ == "__main__":
    main()
