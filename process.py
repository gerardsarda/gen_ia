"""
Analytics Process Functions

This module provides 4 core functions for the analytics pipeline:
1. generate_sql - Generate SQL queries from natural language questions (4 levels)
2. run_query - Execute SQL queries on data
3. explain_results - Explain query results for business stakeholders
4. add_visualization - Create appropriate visualizations based on data types
"""

import pandas as pd
import json
import re
import os
from typing import Optional, Tuple, Dict, Any, List
import traceback

from dotenv import load_dotenv
load_dotenv()


# LLM Provider - Currently using OpenAI, but can be easily switched
# Alternative providers: Anthropic (Claude), Google (Gemini), Azure OpenAI, etc.
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Please install with: pip install openai")

# SQL Execution
try:
    from pandasql import sqldf
    PANDASQL_AVAILABLE = True
except ImportError:
    PANDASQL_AVAILABLE = False
    print("Warning: pandasql not available. Please install with: pip install pandasql")

# Visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Please install with: pip install plotly")

# Configuration
try:
    import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config module not available. Set OPENAI_API_KEY environment variable.")


def _get_schema_info_for_table(df: pd.DataFrame, table_name: str) -> str:
    """Extract schema information for a single table."""
    schema_parts = [f"Table '{table_name}' has {len(df)} rows and {len(df.columns)} columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Simplify dtype names
        if 'int' in dtype:
            dtype_str = 'INTEGER'
        elif 'float' in dtype:
            dtype_str = 'FLOAT'
        elif 'bool' in dtype:
            dtype_str = 'BOOLEAN'
        elif 'datetime' in dtype or 'object' in dtype:
            # Check if it's actually a date
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype_str = 'DATE'
            else:
                dtype_str = 'TEXT'
        else:
            dtype_str = 'TEXT'
        
        schema_parts.append(f"  - {col}: {dtype_str}")
    return '\n'.join(schema_parts)


def _get_schema_info(dfs: Dict[str, pd.DataFrame]) -> str:
    """Extract schema information for multiple tables."""
    schema_parts = []
    for table_name, df in dfs.items():
        schema_parts.append(_get_schema_info_for_table(df, table_name))
        schema_parts.append("")  # Empty line between tables
    return '\n'.join(schema_parts).strip()


def _get_semantics_info_for_table(df: pd.DataFrame, table_name: str) -> str:
    """Extract semantic information for a single table."""
    # Column descriptions for fitlife_members
    members_descriptions = {
        'member_id': "Unique identifier for each member",
        'month': "Month of the record (YYYY-MM format)",
        'center': "Gym center location",
        'plan': "Membership plan type (basic, premium, family)",
        'price_paid': "Amount paid for the month (revenue)",
        'signup_date': "Date when member signed up",
        'acquisition_channel': "How the member was acquired (corporate, online, referral, etc.)",
        'tenure_months': "Number of months since signup",
        'visits_this_month': "Number of gym visits in the month",
        'group_classes_attended': "Number of group classes attended",
        'uses_app': "Whether member uses the mobile app (True/False)",
        'has_personal_trainer': "Whether member has a personal trainer (True/False)",
        'cost_to_serve': "Variable cost to serve this member",
        'status': "Member status (active, churned)",
        'churn_reason': "Reason for churning (if applicable)"
    }
    
    # Column descriptions for fitlife_context
    context_descriptions = {
        'month': "Month of the record (YYYY-MM format) - can be joined with fitlife_members.month",
        'competitor_lowcost_price': "Price of competitor's low-cost option in this month",
        'campaign_active': "Active marketing campaign (january_promo, summer_body, back_to_gym, or empty)",
        'service_incident': "Service incident that occurred (app_outage, equipment_breakdown, heating_failure, or empty)",
        'monthly_fixed_costs': "Fixed operational costs for the month",
        'avg_occupancy_rate': "Average occupancy rate of gym centers (0.0 to 1.0)",
        'acquisition_cost_avg': "Average customer acquisition cost for the month"
    }
    
    # Combine all descriptions
    all_descriptions = {**members_descriptions, **context_descriptions}
    
    semantics_parts = []
    
    # Add table-level description for fitlife_members
    if table_name == 'fitlife_members':
        semantics_parts.append(f"Table '{table_name}' structure:")
        semantics_parts.append("  - Each row represents a user (member) in a given month.")
        semantics_parts.append("  - A member can have multiple rows (one per month they were active).")
        semantics_parts.append("  - The combination of member_id and month uniquely identifies each row.")
        semantics_parts.append("")
    
    semantics_parts.append(f"Table '{table_name}' contains the following columns with semantic descriptions:")
    for col in df.columns:
        desc = all_descriptions.get(col, f"Column {col} (description not available)")
        semantics_parts.append(f"  - {col}: {desc}")
    
    return '\n'.join(semantics_parts)


def _get_semantics_info(dfs: Dict[str, pd.DataFrame]) -> str:
    """Extract semantic information for multiple tables."""
    semantics_parts = []
    for table_name, df in dfs.items():
        semantics_parts.append(_get_semantics_info_for_table(df, table_name))
        semantics_parts.append("")  # Empty line between tables
    
    # Add relationship information
    if 'fitlife_members' in dfs and 'fitlife_context' in dfs:
        semantics_parts.append("Table Relationships:")
        semantics_parts.append("  - fitlife_members.month can be joined with fitlife_context.month")
        semantics_parts.append("  - Both tables share the 'month' column for time-based analysis")
    
    return '\n'.join(semantics_parts).strip()


def _load_golden_dataset(golden_dataset_path: Optional[str] = None) -> List[Dict]:
    """Load golden dataset from JSON file."""
    if golden_dataset_path is None:
        # Try to find it in common locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        golden_dataset_path = os.path.join(current_dir, 'golden_dataset.json')
    
    if os.path.exists(golden_dataset_path):
        try:
            with open(golden_dataset_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load golden dataset: {e}")
            return []
    return []


def save_feedback_to_golden_dataset(
    question: str,
    sql: str,
    is_ok: bool,
    answer: Optional[str] = None,
    golden_dataset_path: Optional[str] = None,
    level: Optional[int] = None
) -> bool:
    """
    Save user feedback to golden dataset JSON file.
    
    Args:
        question: The question that was asked
        sql: The SQL code that was generated
        is_ok: Whether the SQL is correct (True) or incorrect (False)
        answer: Optional full answer/response text
        golden_dataset_path: Path to golden dataset JSON file
        level: Optional level used for SQL generation
    
    Returns:
        True if successfully saved, False otherwise
    """
    from datetime import datetime
    
    if golden_dataset_path is None:
        # Try to find it in common locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        golden_dataset_path = os.path.join(current_dir, 'golden_dataset.json')
    
    # Load existing dataset
    golden_dataset = _load_golden_dataset(golden_dataset_path)
    
    # Create new entry
    new_entry = {
        "question": question,
        "sql_code": sql,
        "is_ok": 1 if is_ok else 0,  # Use 1/0 for compatibility with existing format
        "timestamp": datetime.now().isoformat()
    }
    
    if answer:
        new_entry["answer"] = answer
    
    if level is not None:
        new_entry["level"] = level
    
    # Add to dataset
    golden_dataset.append(new_entry)
    
    # Save back to file
    try:
        with open(golden_dataset_path, 'w') as f:
            json.dump(golden_dataset, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving feedback to golden dataset: {e}")
        return False


def get_user_feedback_interactive() -> Optional[bool]:
    """
    Get user feedback interactively via command line.
    
    Returns:
        True if user says SQL is OK, False if not OK, None if skipped
    """
    try:
        response = input("\nIs the generated SQL correct? (y/n/skip): ").strip().lower()
        if response in ['y', 'yes', '1', 'true']:
            return True
        elif response in ['n', 'no', '0', 'false']:
            return False
        else:
            return None  # Skip
    except (EOFError, KeyboardInterrupt):
        return None  # Skip on Ctrl+C or EOF


def _extract_sql_from_response(response: str) -> Optional[str]:
    """Extract SQL code from LLM response (handles markdown code blocks)."""
    # Remove markdown code blocks
    sql_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # If no code block, try to find SQL-like content
    lines = response.split('\n')
    sql_lines = []
    in_sql = False
    for line in lines:
        if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY']):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
            if line.strip().endswith(';') or (len(sql_lines) > 3 and not line.strip()):
                break
    
    if sql_lines:
        return '\n'.join(sql_lines).strip()
    
    return None


def _extract_table_name_from_sql(sql: str) -> Optional[str]:
    """Extract table name from SQL query."""
    # Look for FROM clause
    from_match = re.search(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    if from_match:
        return from_match.group(1)
    
    # Look for JOIN clauses
    join_match = re.search(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    if join_match:
        return join_match.group(1)
    
    return None


def _call_llm(messages: List[Dict], api_key: Optional[str] = None, model: str = "gpt-4") -> str:
    """
    Call LLM provider to generate response.
    
    Currently uses OpenAI, but can be easily adapted to other providers:
    - Anthropic: from anthropic import Anthropic; client = Anthropic(api_key=api_key)
    - Google: import google.generativeai as genai; genai.configure(api_key=api_key)
    - Azure OpenAI: openai.AzureOpenAI(api_key=api_key, endpoint=endpoint, api_version="2024-02-15-preview")
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not available. Install with: pip install openai")
    
    if api_key is None:
        if CONFIG_AVAILABLE:
            api_key = config.settings.openai_api_key
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or use config.py")
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1  # Lower temperature for more consistent SQL generation
    )
    
    return response.choices[0].message.content


def _create_rag_context(question: str, golden_dataset: List[Dict], max_examples: int = 5) -> str:
    """
    Create RAG context from golden dataset.
    For small datasets, inject directly. For large datasets, use similarity search.
    """
    if len(golden_dataset) <= 20:
        # Small dataset: inject directly in prompt
        positive_examples = [ex for ex in golden_dataset if ex.get('is_ok', 1) == 1]
        examples = positive_examples[-max_examples:]  # Get last N positive examples
        
        rag_context = "\n\nHere are examples of similar questions and their correct SQL solutions:\n\n"
        for i, ex in enumerate(examples, 1):
            sql_code = ex.get('sql_code', '')
            if not sql_code:
                # Try to extract from answer
                sql_code = _extract_sql_from_response(ex.get('answer', ''))
            
            if sql_code:
                rag_context += f"Example {i}:\n"
                rag_context += f"Question: {ex.get('question', '')}\n"
                rag_context += f"SQL:\n```sql\n{sql_code}\n```\n\n"
        
        return rag_context
    else:
        # Large dataset: Use similarity search (simplified - in production use embeddings)
        # For now, just use keyword matching to find similar questions
        question_lower = question.lower()
        keywords = set(question_lower.split())
        
        scored_examples = []
        for ex in golden_dataset:
            if ex.get('is_ok', 1) != 1:
                continue
            
            ex_question = ex.get('question', '').lower()
            ex_keywords = set(ex_question.split())
            
            # Simple keyword overlap score
            overlap = len(keywords & ex_keywords) / max(len(keywords), 1)
            scored_examples.append((overlap, ex))
        
        # Sort by score and take top examples
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        examples = [ex for _, ex in scored_examples[:max_examples]]
        
        rag_context = "\n\nHere are similar questions and their correct SQL solutions:\n\n"
        for i, ex in enumerate(examples, 1):
            sql_code = ex.get('sql_code', '')
            if not sql_code:
                sql_code = _extract_sql_from_response(ex.get('answer', ''))
            
            if sql_code:
                rag_context += f"Example {i}:\n"
                rag_context += f"Question: {ex.get('question', '')}\n"
                rag_context += f"SQL:\n```sql\n{sql_code}\n```\n\n"
        
        return rag_context


def generate_sql(
    question: str,
    df: Optional[pd.DataFrame] = None,
    dfs: Optional[Dict[str, pd.DataFrame]] = None,
    level: int = 3,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    golden_dataset_path: Optional[str] = None,
    negative_examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate SQL query from natural language question.
    The LLM will infer the table name from the context provided at each level.
    
    Args:
        question: Natural language question to convert to SQL
        df: Single DataFrame (for level 1 only, or backward compatibility)
        dfs: Dictionary of DataFrames with table names as keys (required for levels 2-4)
            Expected keys: 'fitlife_members' and 'fitlife_context'
        level: Level of context provided (1-4)
            - Level 1: Pass CSV sample and get query (table name inferred from CSV headers/context)
            - Level 2: Pass only schema (table and column names and formats for both tables)
            - Level 3: Pass schema + semantics (description of tables and fields for both tables)
            - Level 4: Pass schema + semantics + golden dataset (table names from examples)
        api_key: OpenAI API key (optional, uses config or env var if not provided)
        model: LLM model to use (default: gpt-4)
        golden_dataset_path: Path to golden dataset JSON file (for level 4)
        negative_examples: Optional list of dicts with 'question' and 'sql' keys for incorrect examples
    
    Returns:
        SQL query string
    """
    if level < 1 or level > 4:
        raise ValueError("Level must be between 1 and 4")
    
    # For levels 2-4, require dictionary of DataFrames with both tables
    if level >= 2:
        if dfs is None:
            raise ValueError("For levels 2-4, 'dfs' parameter (dict of DataFrames) is required. "
                           "Expected keys: 'fitlife_members' and 'fitlife_context'")
        if 'fitlife_members' not in dfs or 'fitlife_context' not in dfs:
            raise ValueError("For levels 2-4, 'dfs' must contain both 'fitlife_members' and 'fitlife_context'")
    
    # For level 1, can use single DataFrame or dict
    if level == 1:
        if df is not None:
            # Single DataFrame for level 1
            csv_sample = df.head(100).to_csv(index=False)
            context_parts = [f"CSV Data Sample (first 100 rows):\n{csv_sample}"]
        elif dfs is not None:
            # Use first DataFrame from dict for level 1
            first_table = list(dfs.keys())[0]
            csv_sample = dfs[first_table].head(100).to_csv(index=False)
            context_parts = [f"CSV Data Sample (first 100 rows from {first_table}):\n{csv_sample}"]
        else:
            raise ValueError("Either 'df' or 'dfs' parameter is required")
    else:
        # Levels 2-4: use dictionary of DataFrames
        context_parts = []
    
    # Build context based on level
    if level >= 2:
        # Level 2: Include schema for both tables
        schema_info = _get_schema_info(dfs)
        context_parts.append(f"\nSchema Information:\n{schema_info}")
    
    if level >= 3:
        # Level 3: Include semantics for both tables
        semantics_info = _get_semantics_info(dfs)
        context_parts.append(f"\n{semantics_info}")
    
    if level >= 4:
        # Level 4: Include golden dataset - examples will show table names
        golden_dataset = _load_golden_dataset(golden_dataset_path)
        if golden_dataset:
            rag_context = _create_rag_context(question, golden_dataset)
            context_parts.append(rag_context)
    
    # Add negative examples if provided
    if negative_examples:
        negative_context = "\n\nIMPORTANT - Examples of INCORRECT SQL (DO NOT generate similar queries):\n\n"
        for i, neg_ex in enumerate(negative_examples, 1):
            neg_question = neg_ex.get('question', '')
            neg_sql = neg_ex.get('sql', '')
            if neg_question and neg_sql:
                negative_context += f"Incorrect Example {i}:\n"
                negative_context += f"Question: {neg_question}\n"
                negative_context += f"Incorrect SQL (DO NOT use this approach):\n```sql\n{neg_sql}\n```\n\n"
        context_parts.append(negative_context)
    
    # Build prompt
    system_message = """You are a SQL expert. Generate SQL queries that are:
1. Correct and syntactically valid
2. Use only columns that exist in the schema
3. Compatible with SQLite syntax (used by pandasql)
4. Efficient and well-structured
5. Use the appropriate table name(s) as indicated in the schema/semantics/examples
6. When joining tables, use the correct join conditions (e.g., fitlife_members.month = fitlife_context.month)
7. Avoid the mistakes shown in incorrect examples (if any are provided)

Return ONLY the SQL code in a markdown code block, no explanations."""
    
    user_message = f"Question: {question}\n\n"
    user_message += "\n".join(context_parts)
    user_message += "\n\nGenerate SQL code to answer the question. Determine the appropriate table name(s) from the context provided above."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # Call LLM
    response = _call_llm(messages, api_key=api_key, model=model)
    
    # Extract SQL from response
    sql = _extract_sql_from_response(response)
    if not sql:
        raise ValueError(f"Could not extract SQL from LLM response: {response}")
    
    return sql


def run_query(
    sql: str,
    df: Optional[pd.DataFrame] = None,
    dfs: Optional[Dict[str, pd.DataFrame]] = None,
    table_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Execute SQL query on DataFrame(s).
    
    Args:
        sql: SQL query string
        df: Single DataFrame to query (for backward compatibility)
        dfs: Dictionary of DataFrames with table names as keys (for queries with multiple tables)
        table_name: Name of the table in SQL (if None, will be extracted from SQL)
    
    Returns:
        Tuple of (result_dataframe, error_message)
        If successful: (DataFrame, None)
        If error: (None, error_message)
    """
    if not PANDASQL_AVAILABLE:
        return None, "pandasql library not available. Please install it with: pip install pandasql"
    
    # Determine which DataFrames to use
    if dfs is not None:
        # Use dictionary of DataFrames
        dataframes = dfs
    elif df is not None:
        # Extract table name from SQL if not provided
        if table_name is None:
            extracted_table_name = _extract_table_name_from_sql(sql)
            if extracted_table_name:
                table_name = extracted_table_name
            else:
                table_name = 'data'  # Default fallback
        dataframes = {table_name: df}
    else:
        return None, "Either 'df' or 'dfs' parameter is required"
    
    try:
        # Clean up SQL code
        sql = sql.strip()
        
        # Remove markdown code block markers
        sql = re.sub(r'^```.*?\n', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'\n```.*?$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'```', '', sql)
        sql = sql.strip()
        
        # Remove trailing semicolons
        sql = sql.rstrip(';').strip()
        
        # Replace common table name placeholders
        sql = re.sub(r'\bdf\b', 'data', sql, flags=re.IGNORECASE)
        
        # Prepare DataFrames for SQL execution
        dfs_for_sql = {}
        for table_name_key, df_table in dataframes.items():
            df_for_sql = df_table.copy()
            for col in df_for_sql.columns:
                # Convert datetime columns to strings for SQL compatibility
                if pd.api.types.is_datetime64_any_dtype(df_for_sql[col]):
                    if col == 'month':
                        df_for_sql[col] = df_for_sql[col].dt.strftime('%Y-%m')
                    else:
                        df_for_sql[col] = df_for_sql[col].dt.strftime('%Y-%m-%d')
            dfs_for_sql[table_name_key] = df_for_sql
        
        # Execute SQL with all tables available
        result = sqldf(sql, dfs_for_sql)
        
        # Convert result to DataFrame
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return pd.DataFrame(), None
            return result, None
        elif isinstance(result, pd.Series):
            return result.to_frame(), None
        else:
            # Scalar result
            return pd.DataFrame({'result': [result]}), None
            
    except Exception as e:
        error_msg = f"Error executing SQL: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def explain_results(
    question: str,
    sql: str,
    results_df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "gpt-4"
) -> str:
    """
    Explain query results in business-friendly language for stakeholders.
    
    Args:
        question: Original business question
        sql: SQL query that was executed
        results_df: DataFrame with query results
        api_key: OpenAI API key (optional)
        model: LLM model to use (default: gpt-4)
    
    Returns:
        Business-friendly explanation of the results
    """
    # Prepare results summary
    results_summary = f"Query returned {len(results_df)} rows and {len(results_df.columns)} columns.\n\n"
    results_summary += "Column names: " + ", ".join(results_df.columns.tolist()) + "\n\n"
    
    # Add sample data (first 10 rows)
    if len(results_df) > 0:
        results_summary += "Sample results (first 10 rows):\n"
        results_summary += results_df.head(10).to_string(index=False) + "\n\n"
        
        # Add summary statistics for numeric columns
        numeric_cols = results_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            results_summary += "Summary statistics:\n"
            results_summary += results_df[numeric_cols].describe().to_string() + "\n"
    
    system_message = """You are a business analyst explaining data insights to stakeholders.
Your explanations should be:
1. Clear and non-technical
2. Focused on business implications
3. Highlight key findings and trends
4. Use business terminology, not technical jargon
5. Be concise but comprehensive

Explain what the data shows and what it means for the business."""
    
    user_message = f"""Business Question: {question}

SQL Query Used:
```sql
{sql}
```

Query Results:
{results_summary}

Please provide a clear, business-friendly explanation of these results."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    explanation = _call_llm(messages, api_key=api_key, model=model)
    return explanation


def add_visualization(
    results_df: pd.DataFrame,
    question: Optional[str] = None,
    chart_type: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create appropriate visualization based on data types.
    Follows Wall Street Journal Guide to Information Graphics principles:
    - 1 numeric column -> histogram
    - 1 categorical column -> bar chart
    - Date column -> line chart
    - 2 columns (1 categorical, 1 numeric) -> grouped bar chart
    - Time series -> line chart
    
    Args:
        results_df: DataFrame with query results
        question: Original question (optional, for chart title)
        chart_type: Force specific chart type (optional)
    
    Returns:
        Plotly figure object, or None if visualization cannot be created
    """
    if not PLOTLY_AVAILABLE:
        print("Warning: plotly not available. Cannot create visualizations.")
        return None
    
    if len(results_df) == 0:
        print("Warning: Empty results, cannot create visualization.")
        return None
    
    if len(results_df.columns) == 0:
        print("Warning: No columns in results, cannot create visualization.")
        return None
    
    # Determine data types
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = [col for col in results_df.columns if pd.api.types.is_datetime64_any_dtype(results_df[col])]
    
    # If we have a date-like column (month, date, etc.), treat it as datetime
    date_like_cols = [col for col in results_df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time', 'period'])]
    
    # Determine chart type if not specified
    if chart_type is None:
        if len(results_df.columns) == 1:
            col = results_df.columns[0]
            if col in numeric_cols:
                chart_type = 'histogram'
            elif col in categorical_cols or col in date_like_cols:
                chart_type = 'bar'
            else:
                chart_type = 'bar'
        elif len(results_df.columns) == 2:
            col1, col2 = results_df.columns[0], results_df.columns[1]
            # Check if one is date/time
            if col1 in datetime_cols or col1 in date_like_cols:
                chart_type = 'line'
            elif col2 in datetime_cols or col2 in date_like_cols:
                chart_type = 'line'
            # Check if one is categorical and one is numeric
            elif (col1 in categorical_cols and col2 in numeric_cols) or (col2 in categorical_cols and col1 in numeric_cols):
                chart_type = 'bar'
            else:
                chart_type = 'scatter'
        else:
            # Multiple columns - try to find date column for line chart
            if any(col in datetime_cols or col in date_like_cols for col in results_df.columns):
                chart_type = 'line'
            elif len(numeric_cols) >= 2:
                chart_type = 'scatter'
            else:
                chart_type = 'bar'
    
    # Create visualization based on type
    title = question if question else "Query Results"
    
    try:
        if chart_type == 'histogram':
            if len(numeric_cols) > 0:
                fig = px.histogram(
                    results_df,
                    x=numeric_cols[0],
                    title=title,
                    labels={numeric_cols[0]: numeric_cols[0].replace('_', ' ').title()}
                )
                fig.update_layout(
                    xaxis_title=numeric_cols[0].replace('_', ' ').title(),
                    yaxis_title="Frequency",
                    template="plotly_white"
                )
                return fig
        
        elif chart_type == 'bar':
            if len(categorical_cols) > 0:
                # Bar chart of categorical values
                value_col = categorical_cols[0]
                value_counts = results_df[value_col].value_counts().head(20)  # Limit to top 20
                fig = go.Figure(data=[
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        marker_color='steelblue'
                    )
                ])
                fig.update_layout(
                    title=title,
                    xaxis_title=value_col.replace('_', ' ').title(),
                    yaxis_title="Count",
                    template="plotly_white"
                )
                return fig
            elif len(results_df.columns) == 2:
                # Grouped bar chart: categorical x, numeric y
                cat_col = categorical_cols[0] if categorical_cols else results_df.columns[0]
                num_col = numeric_cols[0] if numeric_cols else results_df.columns[1]
                
                fig = px.bar(
                    results_df,
                    x=cat_col,
                    y=num_col,
                    title=title,
                    labels={
                        cat_col: cat_col.replace('_', ' ').title(),
                        num_col: num_col.replace('_', ' ').title()
                    }
                )
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=cat_col.replace('_', ' ').title(),
                    yaxis_title=num_col.replace('_', ' ').title()
                )
                return fig
        
        elif chart_type == 'line':
            # Line chart for time series
            date_col = None
            for col in results_df.columns:
                if col in datetime_cols or col in date_like_cols:
                    date_col = col
                    break
            
            if date_col:
                # Use date column as x-axis
                numeric_col = numeric_cols[0] if numeric_cols else [c for c in results_df.columns if c != date_col][0]
                fig = px.line(
                    results_df,
                    x=date_col,
                    y=numeric_col,
                    title=title,
                    labels={
                        date_col: date_col.replace('_', ' ').title(),
                        numeric_col: numeric_col.replace('_', ' ').title()
                    }
                )
            else:
                # Use first column as x, second as y
                fig = px.line(
                    results_df,
                    x=results_df.columns[0],
                    y=results_df.columns[1],
                    title=title
                )
            
            fig.update_layout(
                template="plotly_white",
                hovermode='x unified'
            )
            return fig
        
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    results_df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=title,
                    labels={
                        numeric_cols[0]: numeric_cols[0].replace('_', ' ').title(),
                        numeric_cols[1]: numeric_cols[1].replace('_', ' ').title()
                    }
                )
                fig.update_layout(template="plotly_white")
                return fig
        
        # Default: table view
        print("Warning: Could not determine appropriate chart type. Consider specifying chart_type.")
        return None
        
    except Exception as e:
        print(f"Error creating visualization: {e}\n{traceback.format_exc()}")
        return None
