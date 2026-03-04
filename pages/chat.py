import os
import sys
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from process import generate_sql, run_query, save_feedback_to_golden_dataset

st.set_page_config(page_title="Data Chat", layout="wide")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_file(filename: str) -> str:
    candidates = [
        os.path.join(_project_root(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


@st.cache_data
def load_dataframes() -> Tuple[Dict[str, pd.DataFrame], str, str]:
    members_path = _find_file("fitlife_members.csv")
    context_path = _find_file("fitlife_context.csv")

    if not members_path or not context_path:
        return {}, members_path, context_path

    members = pd.read_csv(members_path)
    context = pd.read_csv(context_path)

    if "month" in members.columns:
        members["month"] = pd.to_datetime(members["month"], errors="coerce")
    if "signup_date" in members.columns:
        members["signup_date"] = pd.to_datetime(members["signup_date"], errors="coerce")
    if "month" in context.columns:
        context["month"] = pd.to_datetime(context["month"], errors="coerce")

    dfs = {
        "fitlife_members": members,
        "fitlife_context": context,
    }
    return dfs, members_path, context_path


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_sent" not in st.session_state:
        st.session_state.feedback_sent = set()


def _feedback_key(message_id: str, is_ok: bool) -> str:
    return f"{message_id}:{1 if is_ok else 0}"


def render_feedback(message: dict, golden_dataset_path: str) -> None:
    if "id" not in message:
        return

    msg_id = message["id"]
    feedback_row = st.columns([1, 1, 6])

    up_key = f"up_{msg_id}"
    down_key = f"down_{msg_id}"

    with feedback_row[0]:
        if st.button("+1", key=up_key):
            fk = _feedback_key(msg_id, True)
            if fk not in st.session_state.feedback_sent:
                ok = save_feedback_to_golden_dataset(
                    question=message["question"],
                    sql=message["sql"],
                    is_ok=True,
                    answer=message.get("text", ""),
                    golden_dataset_path=golden_dataset_path,
                    level=message.get("level"),
                )
                if ok:
                    st.session_state.feedback_sent.add(fk)
                    st.success("Feedback saved (+1).")
                else:
                    st.error("Could not save feedback.")
            else:
                st.info("Feedback already saved.")

    with feedback_row[1]:
        if st.button("-1", key=down_key):
            fk = _feedback_key(msg_id, False)
            if fk not in st.session_state.feedback_sent:
                ok = save_feedback_to_golden_dataset(
                    question=message["question"],
                    sql=message["sql"],
                    is_ok=False,
                    answer=message.get("text", ""),
                    golden_dataset_path=golden_dataset_path,
                    level=message.get("level"),
                )
                if ok:
                    st.session_state.feedback_sent.add(fk)
                    st.success("Feedback saved (-1).")
                else:
                    st.error("Could not save feedback.")
            else:
                st.info("Feedback already saved.")


def main() -> None:
    st.title("Data Chat")
    st.caption("Ask data questions and get SQL + results. Use +1/-1 to feed the golden dataset.")

    init_state()
    dfs, members_path, context_path = load_dataframes()

    if not dfs:
        st.error("Required CSV files were not found.")
        st.code(f"fitlife_members.csv: {members_path or 'not found'}")
        st.code(f"fitlife_context.csv: {context_path or 'not found'}")
        return

    root = _project_root()
    golden_dataset_path = os.path.join(root, "golden_dataset.json")

    with st.sidebar:
        st.subheader("Config")
        model = st.text_input("OpenAI model", value="gpt-4")
        level = st.selectbox("Context level", [1, 2, 3, 4], index=3)
        st.caption(f"Golden dataset: {golden_dataset_path}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message["role"] == "assistant":
                if message.get("sql"):
                    st.code(message["sql"], language="sql")
                if message.get("result") is not None:
                    st.dataframe(message["result"], use_container_width=True)
                    st.caption(f"Rows returned: {len(message['result'])}")
                render_feedback(message, golden_dataset_path)

    question = st.chat_input("Ask a question about your data...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "text": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and running query..."):
            try:
                sql = generate_sql(
                    question=question,
                    dfs=dfs,
                    level=level,
                    model=model,
                    golden_dataset_path=golden_dataset_path,
                )
                result_df, err = run_query(sql=sql, dfs=dfs)

                if err:
                    assistant_text = f"Could not run query. Error: `{err}`"
                    st.markdown(assistant_text)
                    st.code(sql, language="sql")
                    msg_id = str(len(st.session_state.messages))
                    st.session_state.messages.append(
                        {
                            "id": msg_id,
                            "role": "assistant",
                            "text": assistant_text,
                            "question": question,
                            "sql": sql,
                            "result": None,
                            "level": level,
                        }
                    )
                    render_feedback(st.session_state.messages[-1], golden_dataset_path)
                    return

                assistant_text = "Query executed successfully."
                st.markdown(assistant_text)
                st.code(sql, language="sql")
                st.dataframe(result_df, use_container_width=True)
                st.caption(f"Rows returned: {len(result_df)}")

                msg_id = str(len(st.session_state.messages))
                st.session_state.messages.append(
                    {
                        "id": msg_id,
                        "role": "assistant",
                        "text": assistant_text,
                        "question": question,
                        "sql": sql,
                        "result": result_df,
                        "level": level,
                    }
                )
                render_feedback(st.session_state.messages[-1], golden_dataset_path)

            except Exception as exc:
                err_text = f"Error generating response: `{exc}`"
                st.markdown(err_text)
                msg_id = str(len(st.session_state.messages))
                st.session_state.messages.append(
                    {
                        "id": msg_id,
                        "role": "assistant",
                        "text": err_text,
                        "question": question,
                        "sql": "",
                        "result": None,
                        "level": level,
                    }
                )


if __name__ == "__main__":
    main()
