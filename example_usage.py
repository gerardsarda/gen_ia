"""
Example usage of the analytics process functions.

This script demonstrates how to use:
1. generate_sql - Generate SQL from natural language
2. run_query - Execute SQL queries
3. explain_results - Explain results for business stakeholders
4. add_visualization - Create appropriate visualizations
"""

import pandas as pd
from process import (
    generate_sql, 
    run_query, 
    explain_results, 
    add_visualization,
    save_feedback_to_golden_dataset,
    get_user_feedback_interactive
)
import os

# Load data
def load_data():
    """Load the Fitlife members and context data."""
    members_path = os.path.join(os.path.dirname(__file__), 'fitlife_members.csv')
    context_path = os.path.join(os.path.dirname(__file__), 'fitlife_context.csv')
    
    members = pd.read_csv(members_path)
    members['month'] = pd.to_datetime(members['month'], format='%Y-%m')
    members['signup_date'] = pd.to_datetime(members['signup_date'])
    
    context = pd.read_csv(context_path)
    context['month'] = pd.to_datetime(context['month'], format='%Y-%m')
    
    return members, context

def main():
    """Example usage of all 4 functions."""
    
    # Load data
    print("Loading data...")
    members_df, context_df = load_data()
    print(f"Loaded {len(members_df)} member rows and {len(context_df)} context rows")
    
    import random

    # Load list of questions from BUSINESS_QUESTIONS.md and pick one at random
    def load_business_questions():
        import re
        questions = []
        md_path = os.path.join(os.path.dirname(__file__), "BUSINESS_QUESTIONS.md")
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                # Match lines that start with a number followed by a period (e.g., "1. **Question?**")
                # Pattern: number. **question text** (optional context)
                match = re.match(r'^\d+\.\s+\*\*(.+?)\*\*(?:\s*\(.+?\))?$', line_strip)
                if match:
                    # Extract just the question text (remove markdown formatting)
                    question_text = match.group(1).strip()
                    questions.append(question_text)
        return questions

    business_questions = load_business_questions()
    question = random.choice(business_questions) if business_questions else "What is the total revenue by plan type?"
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Step 1: Generate SQL (using different levels)
    print("Step 1: Generating SQL query...")
    print("Using Level 3 (Schema + Semantics with both tables)...")
    
    try:
        sql = generate_sql(
            question=question,
            dfs={
                'fitlife_members': members_df,
                'fitlife_context': context_df
            },
            level=3,  # Use level 3: schema + semantics (both tables included)
            # api_key='your-api-key-here',  # Optional, uses config or env var
            model='gpt-4'  # or 'gpt-3.5-turbo' for faster/cheaper
        )
        print(f"Generated SQL:\n{sql}\n")
        
        # Collect user feedback on the generated SQL
        print("\n" + "="*60)
        feedback = get_user_feedback_interactive()
        
        if feedback is None:
            # User skipped feedback - don't execute
            print("Feedback skipped. Stopping execution.")
            print("="*60 + "\n")
            return
        elif feedback is False:
            # SQL is incorrect - save feedback and retry with negative example
            success = save_feedback_to_golden_dataset(
                question=question,
                sql=sql,
                is_ok=False,
                answer=f"Generated SQL for question: {question}",
                level=3
            )
            if success:
                print("Feedback saved to golden_dataset.json (✗ Incorrect)")
            else:
                print("Warning: Could not save feedback to golden dataset")
            
            # Retry SQL generation with the incorrect SQL as a negative example
            print("\nRetrying SQL generation with feedback...")
            try:
                sql = generate_sql(
                    question=question,
                    dfs={
                        'fitlife_members': members_df,
                        'fitlife_context': context_df
                    },
                    level=3,
                    # api_key='your-api-key-here',  # Optional, uses config or env var
                    model='gpt-4',
                    negative_examples=[{
                        'question': question,
                        'sql': sql  # The incorrect SQL as a negative example
                    }]
                )
                print(f"Regenerated SQL:\n{sql}\n")
                
                # Ask for feedback again on the regenerated SQL
                print("="*60)
                feedback_retry = get_user_feedback_interactive()
                
                if feedback_retry is None:
                    print("Feedback skipped. Stopping execution.")
                    print("="*60 + "\n")
                    return
                elif feedback_retry is False:
                    # Still incorrect - save and stop
                    save_feedback_to_golden_dataset(
                        question=question,
                        sql=sql,
                        is_ok=False,
                        answer=f"Regenerated SQL for question: {question}",
                        level=3
                    )
                    print("SQL still marked as incorrect. Stopping execution.")
                    print("="*60 + "\n")
                    return
                else:
                    # Now correct - save and continue
                    save_feedback_to_golden_dataset(
                        question=question,
                        sql=sql,
                        is_ok=True,
                        answer=f"Regenerated SQL for question: {question}",
                        level=3
                    )
                    print("Feedback saved to golden_dataset.json (✓ Correct)")
                    print("SQL approved. Continuing with execution...")
                    print("="*60 + "\n")
            except Exception as e:
                print(f"Error regenerating SQL: {e}")
                print("Stopping execution.")
                print("="*60 + "\n")
                return
        else:
            # SQL is correct - save feedback and continue
            success = save_feedback_to_golden_dataset(
                question=question,
                sql=sql,
                is_ok=True,
                answer=f"Generated SQL for question: {question}",
                level=3
            )
            if success:
                print("Feedback saved to golden_dataset.json (✓ Correct)")
            else:
                print("Warning: Could not save feedback to golden dataset")
            print("SQL approved. Continuing with execution...")
            print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return
    
    # Step 2: Run the query (only if feedback was positive)
    print("Step 2: Executing SQL query...")
    # Pass both tables as dictionary
    results_df, error = run_query(
        sql,
        dfs={
            'fitlife_members': members_df,
            'fitlife_context': context_df
        }
    )
    
    if error:
        print(f"Error: {error}")
        return
    
    print(f"Query returned {len(results_df)} rows")
    print(f"\nResults:\n{results_df}\n")
    
    # Step 3: Explain results
    print("Step 3: Explaining results for business stakeholders...")
    try:
        explanation = explain_results(
            question=question,
            sql=sql,
            results_df=results_df,
            # api_key='your-api-key-here',  # Optional
            model='gpt-4o.mini'
        )
        print(f"\nBusiness Explanation:\n{explanation}\n")
    except Exception as e:
        print(f"Error explaining results: {e}")
    
    # Step 4: Add visualization
    print("Step 4: Creating visualization...")
    fig = add_visualization(
        results_df=results_df,
        question=question
    )
    
    if fig:
        print("Visualization created successfully!")
        # In a Streamlit app, you would use: st.plotly_chart(fig)
        # For this example, we'll save it as PNG
        try:
            fig.write_image("example_visualization.png", width=1200, height=800, scale=2)
            print("Visualization saved to example_visualization.png")
        except Exception as e:
            print(f"Warning: Could not save as PNG: {e}")
            print("Note: PNG export requires 'kaleido' package. Install with: pip install kaleido")
            # Fallback to HTML
            fig.write_html("example_visualization.html")
            print("Saved as HTML instead: example_visualization.html")
    else:
        print("Could not create visualization")
    
    print(f"\n{'='*60}")
    print("Example completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
