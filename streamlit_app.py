######################################################################################################################################################
######
######
###### Sesión 1. 25/02/2026
######
######
###### Las preguntas que sacamos entre todos son:
######  - Financials: <---- este bloque es el que contiene este dashboard
######      - Margent
######      - Ingresos
######      - Costes
######      - Pricing
######      - LTV & CAC
######  - Cliente
######      - KPIs principales:
######          - Activos
######          - Visitas
######      - Comportamiento:
######          - Entrenamiento personal/clases
######          - Movilidad de plan
######      - Churn:
######          - Causas
######          - Campaña/Incidencia
######  
# Filtros:
######      - Mes/Periodo
######      - Centro
######      - Plan
######  
######
###### CONSIDERACIONES:
######      - Estacionalidad/Evolución temporal de las variables
######      - Poner en context los KPIs:
######          - Comparación vs periodos previos
######          - Ratios vs unidad de ingresos (en este caso miembros activos)
######
######
######################################################################################################################################################





import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    """Load members and context data."""
    # build paths relative to this file. it's safer to assume the CSVs live next to
    # the module, especially given our nested `gen_ia/gen_ia` layout in development
    base_dir = os.path.dirname(__file__)  # same folder as streamlit_app.py
    members_path = os.path.join(base_dir, 'fitlife_members.csv')
    context_path = os.path.join(base_dir, 'fitlife_context.csv')

    # debug help: if the application is deployed and the files are missing we'll
    # show the paths being checked so you can adjust the repo or move the files.
    if not os.path.exists(members_path) or not os.path.exists(context_path):
        st.error("Required data files not found.\n"
                 f"Looking for:\n  {members_path}\n  {context_path}")
        # return empty frames to avoid later crashes
        return pd.DataFrame(), pd.DataFrame()

    members = pd.read_csv(members_path)
    members['month'] = pd.to_datetime(members['month'], format='%Y-%m')
    members['signup_date'] = pd.to_datetime(members['signup_date'])
    
    context = pd.read_csv(context_path)
    context['month'] = pd.to_datetime(context['month'], format='%Y-%m')
    
    return members, context

def calculate_ltv(members_df, member_id):
    """Calculate Lifetime Value for a member."""
    member_data = members_df[members_df['member_id'] == member_id]
    if len(member_data) == 0:
        return 0
    total_revenue = member_data['price_paid'].sum()
    total_costs = member_data['cost_to_serve'].sum() if 'cost_to_serve' in member_data.columns else 0
    return total_revenue - total_costs

def main():
    st.title("💰 Financials Dashboard")
    st.markdown("Comprehensive financial analysis with interactive dashboards")
    
    # Load data
    members, context = load_data()
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Month selector
    available_months = sorted(members['month'].unique(), reverse=True)
    month_options = [m.strftime('%Y-%m') for m in available_months]
    selected_month_str = st.sidebar.selectbox("Select Month", month_options, index=0)
    target_month = pd.to_datetime(selected_month_str)
    
    # Center filter
    available_centers = ['All'] + sorted(members['center'].unique().tolist())
    selected_center = st.sidebar.selectbox("Filter by Center", available_centers)
    
    # Plan filter
    available_plans = ['All'] + sorted(members['plan'].unique().tolist())
    selected_plan = st.sidebar.selectbox("Filter by Plan", available_plans)
    
    # ============================================================================
    # DETERMINE ANALYSIS PERIOD BASED ON COMPARISON TYPE
    # ============================================================================
    # Add comparison selector first to determine period
    comparison_type = st.sidebar.selectbox(
        "Comparison Type",
        ["None", "MoM (Month-over-Month)", "QoQ (Quarter-over-Quarter)", "YoY (Year-over-Year)"],
        index=1,  # Default to MoM (Month-over-Month)
        key="comparison_type"
    )
    
    # Determine the period to analyze
    # The main period is always based on the selected month and comparison type
    if comparison_type == "MoM (Month-over-Month)":
        # Analyze selected month only (comparison will be previous month)
        period_months = [target_month]
        period_label = f"for {selected_month_str} (MoM)"
    elif comparison_type == "QoQ (Quarter-over-Quarter)":
        # Analyze last 3 months including current month: [current, current-1, current-2]
        # For 2024-12: [2024-12, 2024-11, 2024-10]
        period_months = [target_month - pd.DateOffset(months=i) for i in range(0, 3)]
        period_label = f"for last 3 months ending {selected_month_str} (QoQ)"
    elif comparison_type == "YoY (Year-over-Year)":
        # Analyze last 12 months including current month
        period_months = [target_month - pd.DateOffset(months=i) for i in range(0, 12)]
        period_label = f"for last 12 months ending {selected_month_str} (YoY)"
    else:
        # Analyze selected month only
        period_months = [target_month]
        period_label = f"for {selected_month_str}"
    
    # Get all data for the analysis period (both active and churned)
    period_data = members[members['month'].isin(period_months)].copy()
    if selected_center != 'All':
        period_data = period_data[period_data['center'] == selected_center]
    if selected_plan != 'All':
        period_data = period_data[period_data['plan'] == selected_plan]
    
    # Get active members data for revenue analysis
    active_data = period_data[period_data['status'] == 'active'].copy()
    
    # Calculate total distinct members in the period (active + churned)
    total_members_in_period = period_data['member_id'].nunique()
    
    # For each member, get their latest status in the period (to handle cases where they might have both active and churned records)
    member_status = period_data.sort_values('month').groupby('member_id')['status'].last()
    
    # Count members who are churned (based on their latest status in the period)
    churned_in_period = (member_status == 'churned').sum()
    
    # Active members = Total - Churned
    active_members = total_members_in_period - churned_in_period
    
    # Calculate revenue and costs for ALL members in the period (both active and churned)
    total_revenue = period_data['price_paid'].sum()
    variable_costs = period_data['cost_to_serve'].sum() if 'cost_to_serve' in period_data.columns else 0
    
    # Fixed costs: sum for all months in the period
    fixed_costs = 0
    for month in period_months:
        fixed_costs_row = context[context['month'] == month]
        if len(fixed_costs_row) > 0 and 'monthly_fixed_costs' in fixed_costs_row.columns:
            fixed_costs += fixed_costs_row['monthly_fixed_costs'].iloc[0]
    
    # Adjust fixed costs based on filters
    # If filtering by center: divide by number of centers (assumed to be 5)
    NUM_CENTERS = 5  # Assumed total number of centers
    if selected_center != 'All':
        fixed_costs = fixed_costs / NUM_CENTERS
    
    # If filtering by plan: allocate proportionally based on members
    if selected_plan != 'All':
        # Get total members across all plans for the period (unfiltered by plan)
        all_plans_data = members[members['month'].isin(period_months)].copy()
        if selected_center != 'All':
            all_plans_data = all_plans_data[all_plans_data['center'] == selected_center]
        total_members_all_plans = all_plans_data['member_id'].nunique()
        
        # Get members in the selected plan for the period
        members_in_selected_plan = period_data['member_id'].nunique()
        
        # Allocate fixed costs proportionally
        if total_members_all_plans > 0:
            fixed_costs = fixed_costs * (members_in_selected_plan / total_members_all_plans)
    
    contribution_margin = total_revenue - variable_costs
    contribution_margin_pct = (contribution_margin / total_revenue * 100) if total_revenue > 0 else 0
    variable_costs_pct = (variable_costs / total_revenue * 100) if total_revenue > 0 else 0
    fixed_costs_pct = (fixed_costs / total_revenue * 100) if total_revenue > 0 else 0
    net_result = contribution_margin - fixed_costs
    net_result_pct = (net_result / total_revenue * 100) if total_revenue > 0 else 0
    
    # Calculate per-member metrics (using total_members_in_period, not just active)
    revenue_per_member = total_revenue / total_members_in_period if total_members_in_period > 0 else 0
    variable_cost_per_member = variable_costs / total_members_in_period if total_members_in_period > 0 else 0
    contribution_per_member = contribution_margin / total_members_in_period if total_members_in_period > 0 else 0
    contribution_per_member_pct = contribution_margin_pct  # Same percentage
    fixed_cost_per_member = fixed_costs / total_members_in_period if total_members_in_period > 0 else 0
    net_result_per_member = net_result / total_members_in_period if total_members_in_period > 0 else 0
    net_result_per_member_pct = net_result_pct  # Same percentage
    
    # ============================================================================
    # COMPARISON CALCULATIONS (for delta display)
    # ============================================================================
    # Calculate comparison period for delta calculations
    comparison_month = None
    comparison_label = ""
    if comparison_type != "None":
        if comparison_type == "MoM (Month-over-Month)":
            # Previous month
            comparison_month = target_month - pd.DateOffset(months=1)
            comparison_label = "MoM"
        elif comparison_type == "QoQ (Quarter-over-Quarter)":
            # Previous quarter (3 months ago)
            comparison_month = target_month - pd.DateOffset(months=3)
            comparison_label = "QoQ"
        elif comparison_type == "YoY (Year-over-Year)":
            # Same month, previous year
            comparison_month = target_month - pd.DateOffset(years=1)
            comparison_label = "YoY"
    
    # Calculate comparison metrics if comparison is selected
    comp_total_members_in_period = None
    comp_churned_in_period = None
    comp_active_members = None
    comp_total_revenue = None
    comp_variable_costs = None
    comp_variable_costs_pct = None
    comp_fixed_costs = None
    comp_fixed_costs_pct = None
    comp_contribution_margin = None
    comp_contribution_margin_pct = None
    comp_net_result = None
    comp_net_result_pct = None
    comp_revenue_per_member = None
    comp_variable_cost_per_member = None
    comp_contribution_per_member = None
    comp_fixed_cost_per_member = None
    comp_net_result_per_member = None
    
    if comparison_month is not None:
        # Determine comparison period months (same structure as main period)
        if comparison_type == "MoM (Month-over-Month)":
            # Comparison is the previous month
            comp_period_months = [comparison_month]
        elif comparison_type == "QoQ (Quarter-over-Quarter)":
            # Comparison is the previous quarter: 3 months ending at comparison_month
            # For comparison_month = 2024-09: [2024-09, 2024-08, 2024-07]
            comp_period_months = [comparison_month - pd.DateOffset(months=i) for i in range(0, 3)]
        elif comparison_type == "YoY (Year-over-Year)":
            # Comparison is the previous year: 12 months ending at comparison_month
            comp_period_months = [comparison_month - pd.DateOffset(months=i) for i in range(0, 12)]
        else:
            comp_period_months = [comparison_month]
        
        # Get all data for comparison period
        comp_period_data = members[members['month'].isin(comp_period_months)].copy()
        if selected_center != 'All':
            comp_period_data = comp_period_data[comp_period_data['center'] == selected_center]
        if selected_plan != 'All':
            comp_period_data = comp_period_data[comp_period_data['plan'] == selected_plan]
        
        # Calculate total distinct members in comparison period (active + churned)
        comp_total_members_in_period = comp_period_data['member_id'].nunique()
        
        # For each member, get their latest status in the comparison period
        comp_member_status = comp_period_data.sort_values('month').groupby('member_id')['status'].last()
        
        # Count members who are churned (based on their latest status in the period)
        comp_churned_in_period = (comp_member_status == 'churned').sum()
        
        # Active members = Total - Churned
        comp_active_members = comp_total_members_in_period - comp_churned_in_period
        
        # Calculate revenue and costs for ALL members in comparison period
        comp_total_revenue = comp_period_data['price_paid'].sum()
        comp_variable_costs = comp_period_data['cost_to_serve'].sum() if 'cost_to_serve' in comp_period_data.columns else 0
        
        # Fixed costs: sum for all months in comparison period
        comp_fixed_costs = 0
        for month in comp_period_months:
            comp_fixed_costs_row = context[context['month'] == month]
            if len(comp_fixed_costs_row) > 0 and 'monthly_fixed_costs' in comp_fixed_costs_row.columns:
                comp_fixed_costs += comp_fixed_costs_row['monthly_fixed_costs'].iloc[0]
        
        # Adjust comparison fixed costs based on filters (same logic as main period)
        # If filtering by center: divide by number of centers (assumed to be 5)
        NUM_CENTERS = 5  # Assumed total number of centers
        if selected_center != 'All':
            comp_fixed_costs = comp_fixed_costs / NUM_CENTERS
        
        # If filtering by plan: allocate proportionally based on members
        if selected_plan != 'All':
            # Get total members across all plans for the comparison period (unfiltered by plan)
            comp_all_plans_data = members[members['month'].isin(comp_period_months)].copy()
            if selected_center != 'All':
                comp_all_plans_data = comp_all_plans_data[comp_all_plans_data['center'] == selected_center]
            comp_total_members_all_plans = comp_all_plans_data['member_id'].nunique()
            
            # Get members in the selected plan for the comparison period
            comp_members_in_selected_plan = comp_period_data['member_id'].nunique()
            
            # Allocate fixed costs proportionally
            if comp_total_members_all_plans > 0:
                comp_fixed_costs = comp_fixed_costs * (comp_members_in_selected_plan / comp_total_members_all_plans)
        
        comp_contribution_margin = comp_total_revenue - comp_variable_costs
        comp_net_result = comp_contribution_margin - comp_fixed_costs
        
        # Calculate percentages for comparison period
        comp_contribution_margin_pct = (comp_contribution_margin / comp_total_revenue * 100) if comp_total_revenue > 0 else 0
        comp_variable_costs_pct = (comp_variable_costs / comp_total_revenue * 100) if comp_total_revenue > 0 else 0
        comp_fixed_costs_pct = (comp_fixed_costs / comp_total_revenue * 100) if comp_total_revenue > 0 else 0
        comp_net_result_pct = (comp_net_result / comp_total_revenue * 100) if comp_total_revenue > 0 else 0
        
        # Per member metrics (using total_members_in_period)
        comp_revenue_per_member = comp_total_revenue / comp_total_members_in_period if comp_total_members_in_period > 0 else 0
        comp_variable_cost_per_member = comp_variable_costs / comp_total_members_in_period if comp_total_members_in_period > 0 else 0
        comp_contribution_per_member = comp_contribution_margin / comp_total_members_in_period if comp_total_members_in_period > 0 else 0
        comp_contribution_per_member_pct = comp_contribution_margin_pct
        comp_fixed_cost_per_member = comp_fixed_costs / comp_total_members_in_period if comp_total_members_in_period > 0 else 0
        comp_net_result_per_member = comp_net_result / comp_total_members_in_period if comp_total_members_in_period > 0 else 0
        comp_net_result_per_member_pct = comp_net_result_pct
    
    # Helper function to calculate delta
    def calc_delta(current, previous):
        if previous is None or previous == 0:
            return None
        return ((current - previous) / previous) * 100
    
    # Helper function to format delta for display
    def format_delta(current, previous, is_percentage=False):
        if previous is None:
            return None
        delta_pct = calc_delta(current, previous)
        if delta_pct is None:
            return None
        if is_percentage:
            # For percentages, show absolute difference
            return f"{delta_pct:+.1f}pp"
        else:
            return f"{delta_pct:+.1f}%"
    
    # ============================================================================
    # UNIT ECONOMICS SECTION - CLEAR DISPLAY
    # ============================================================================
    # Note: We always show current period metrics, with deltas from comparison if selected
    # The period analyzed depends on comparison type (MoM = 1 month, QoQ = 3 months, YoY = 12 months)
    
    st.header(f"💼 Unit Economics {period_label}")
    if comparison_type != "None":
        st.caption(f"Showing metrics for the analyzed period. Deltas show change vs previous period.")
    
    # Total View
    st.subheader("📊 Total")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        delta_members = format_delta(total_members_in_period, comp_total_members_in_period) if comp_total_members_in_period is not None else None
        st.metric("Total Members", f"{total_members_in_period:,}", 
                 delta=delta_members if delta_members else None)
        st.caption("(Active + Churned)")
    
    with col2:
        delta_revenue = format_delta(total_revenue, comp_total_revenue) if comp_total_revenue is not None else None
        st.metric("Revenue", f"€{total_revenue:,.0f}",
                 delta=delta_revenue if delta_revenue else None)
        st.caption(f"Avg: €{revenue_per_member:.2f}")
    
    with col3:
        delta_var_costs = format_delta(variable_costs, comp_variable_costs) if comp_variable_costs is not None else None
        delta_var_costs_pct = format_delta(variable_costs_pct, comp_variable_costs_pct, is_percentage=True) if comp_variable_costs_pct is not None else None
        st.metric("Variable Costs", f"€{variable_costs:,.0f}",
                 delta=delta_var_costs if delta_var_costs else None)
        caption_var = f"**{variable_costs_pct:.1f}%** of Revenue"
        if delta_var_costs_pct and comparison_label:
            caption_var += f" ({delta_var_costs_pct} {comparison_label})"
        st.caption(caption_var)
    
    with col4:
        delta_contrib = format_delta(contribution_margin, comp_contribution_margin) if comp_contribution_margin is not None else None
        delta_contrib_pct = format_delta(contribution_margin_pct, comp_contribution_margin_pct, is_percentage=True) if comp_contribution_margin_pct is not None else None
        st.metric("Contribution Margin", f"€{contribution_margin:,.0f}",
                 delta=delta_contrib if delta_contrib else None)
        caption_contrib = f"**{contribution_margin_pct:.1f}%** of Revenue"
        if delta_contrib_pct and comparison_label:
            caption_contrib += f" ({delta_contrib_pct} {comparison_label})"
        st.caption(caption_contrib)
    
    with col5:
        delta_fixed = format_delta(fixed_costs, comp_fixed_costs) if comp_fixed_costs is not None else None
        delta_fixed_pct = format_delta(fixed_costs_pct, comp_fixed_costs_pct, is_percentage=True) if comp_fixed_costs_pct is not None else None
        st.metric("Fixed Costs", f"€{fixed_costs:,.0f}",
                 delta=delta_fixed if delta_fixed else None)
        caption_fixed = f"**{fixed_costs_pct:.1f}%** of Revenue"
        if delta_fixed_pct and comparison_label:
            caption_fixed += f" ({delta_fixed_pct} {comparison_label})"
        st.caption(caption_fixed)
    
    with col6:
        delta_net = format_delta(net_result, comp_net_result) if comp_net_result is not None else None
        delta_net_pct = format_delta(net_result_pct, comp_net_result_pct, is_percentage=True) if comp_net_result_pct is not None else None
        st.metric("Net Result", f"€{net_result:,.0f}",
                 delta=delta_net if delta_net else f"{net_result_pct:.1f}%",
                 delta_color="inverse" if net_result < 0 else "normal")
        caption_net = f"**{net_result_pct:.1f}%** of Revenue"
        if delta_net_pct and comparison_label:
            caption_net += f" ({delta_net_pct} {comparison_label})"
        st.caption(caption_net)
    
    # Second row: Churned and Active Members
    col1, col2, _,_,_,_ = st.columns(6)
    with col1:
        delta_churned = format_delta(churned_in_period, comp_churned_in_period) if comp_churned_in_period is not None else None
        st.metric("Churned", f"{churned_in_period:,}",
                 delta=delta_churned if delta_churned else None)
    
    with col2:
        delta_active = format_delta(active_members, comp_active_members) if comp_active_members is not None else None
        st.metric("Active Members (End of Month)", f"{active_members:,}",
                 delta=delta_active if delta_active else None, help= "Total - Churned")
        #st.caption("(Total - Churned)")
    
    st.divider()
    
    # Per Member View
    st.subheader("👤 per Member")
    _, col2, col3, col4, col5, col6 = st.columns(6)
    

    
    with col2:
        delta_rev_pm = format_delta(revenue_per_member, comp_revenue_per_member) if comp_revenue_per_member is not None else None
        st.metric("Revenue", f"€{revenue_per_member:.2f}",
                 delta=delta_rev_pm if delta_rev_pm else None)
        st.caption("(Avg Price Paid)")
    
    with col3:
        delta_var_pm = format_delta(variable_cost_per_member, comp_variable_cost_per_member) if comp_variable_cost_per_member is not None else None
        st.metric("Variable Costs", f"€{variable_cost_per_member:.2f}",
                 delta=delta_var_pm if delta_var_pm else None)
        st.caption("(Cost to Serve)")
    
    with col4:
        delta_contrib_pm = format_delta(contribution_per_member, comp_contribution_per_member) if comp_contribution_per_member is not None else None
        st.metric("Contribution Margin", f"€{contribution_per_member:.2f}",
                 delta=delta_contrib_pm if delta_contrib_pm else None)
        st.caption(f"**{contribution_per_member_pct:.1f}%** margin")
    
    with col5:
        delta_fixed_pm = format_delta(fixed_cost_per_member, comp_fixed_cost_per_member) if comp_fixed_cost_per_member is not None else None
        st.metric("Fixed Costs", f"€{fixed_cost_per_member:.2f}",
                 delta=delta_fixed_pm if delta_fixed_pm else None)
    
    with col6:
        delta_net_pm = format_delta(net_result_per_member, comp_net_result_per_member) if comp_net_result_per_member is not None else None
        st.metric("Net Result", f"€{net_result_per_member:.2f}",
                 delta=delta_net_pm if delta_net_pm else f"{net_result_per_member_pct:.1f}%",
                 delta_color="inverse" if net_result_per_member < 0 else "normal")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💰 P&L Overview", "📉 CAC", "📊 Margins"])
    
    # ============================================================================
    # TAB 1: Revenue & Sources
    # ============================================================================
    with tab1:
        st.header("💰 Revenue & Sources")
        
        # Temporal evolution of big numbers
        st.subheader("📈 Big Numbers Evolution Over Time")
        
        # Calculate metrics over time with filters
        temporal_data = members.copy()
        if selected_center != 'All':
            temporal_data = temporal_data[temporal_data['center'] == selected_center]
        if selected_plan != 'All':
            temporal_data = temporal_data[temporal_data['plan'] == selected_plan]
        
        # Calculate revenue and variable costs for ALL members (active + churned) over time
        # This matches the unit economics calculation logic
        metrics_over_time = temporal_data.groupby('month').agg({
            'price_paid': 'sum',
            'cost_to_serve': 'sum',
            'member_id': 'nunique'
        }).reset_index()
        metrics_over_time.columns = ['Month', 'Revenue', 'Variable Costs', 'Total Members']
        metrics_over_time = metrics_over_time.sort_values('Month')
        
        # Calculate active members separately
        active_over_time = temporal_data[temporal_data['status'] == 'active'].groupby('month').agg({
            'member_id': 'nunique'
        }).reset_index()
        active_over_time.columns = ['Month', 'Active Members']
        metrics_over_time = pd.merge(metrics_over_time, active_over_time, on='Month', how='left')
        metrics_over_time['Active Members'] = metrics_over_time['Active Members'].fillna(0)
        
        # Merge with context to get fixed costs
        if 'monthly_fixed_costs' in context.columns:
            fixed_costs_time = context[['month', 'monthly_fixed_costs']].copy()
            fixed_costs_time.columns = ['Month', 'Fixed Costs']
            metrics_over_time = pd.merge(metrics_over_time, fixed_costs_time, on='Month', how='left')
            metrics_over_time['Fixed Costs'] = metrics_over_time['Fixed Costs'].fillna(0)
            
            # Adjust fixed costs based on filters (same logic as unit economics)
            # If filtering by center: divide by number of centers (assumed to be 5)
            NUM_CENTERS = 5  # Assumed total number of centers
            if selected_center != 'All':
                metrics_over_time['Fixed Costs'] = metrics_over_time['Fixed Costs'] / NUM_CENTERS
            
            # If filtering by plan: allocate proportionally based on members
            if selected_plan != 'All':
                # For each month, calculate proportional allocation
                for idx, row in metrics_over_time.iterrows():
                    month = row['Month']
                    # Get total members across all plans for this month (unfiltered by plan)
                    month_all_plans_data = temporal_data[temporal_data['month'] == month].copy()
                    if selected_center != 'All':
                        month_all_plans_data = month_all_plans_data[month_all_plans_data['center'] == selected_center]
                    month_total_members_all_plans = month_all_plans_data['member_id'].nunique()
                    
                    # Get members in the selected plan for this month
                    month_selected_plan_data = temporal_data[(temporal_data['month'] == month) & (temporal_data['plan'] == selected_plan)].copy()
                    if selected_center != 'All':
                        month_selected_plan_data = month_selected_plan_data[month_selected_plan_data['center'] == selected_center]
                    month_members_in_selected_plan = month_selected_plan_data['member_id'].nunique()
                    
                    # Allocate fixed costs proportionally
                    if month_total_members_all_plans > 0:
                        metrics_over_time.at[idx, 'Fixed Costs'] = metrics_over_time.at[idx, 'Fixed Costs'] * (month_members_in_selected_plan / month_total_members_all_plans)
        else:
            metrics_over_time['Fixed Costs'] = 0
        
        # Calculate total costs and profit
        metrics_over_time['Total Costs'] = metrics_over_time['Variable Costs'] + metrics_over_time['Fixed Costs']
        metrics_over_time['Profit'] = metrics_over_time['Revenue'] - metrics_over_time['Total Costs']
        
        # Display charts: Revenue/Costs/Profit combined, and Active Members
        col1, col2 = st.columns(2)
        with col1:
            # Combined Revenue, Costs, and Profit chart
            fig_combined = go.Figure()
            
            # Format numbers for hover (2 decimals, show in k format for values >= 1000)
            def format_hover_value(val):
                if abs(val) >= 1000:
                    return f"€{val/1000:.2f}k"
                else:
                    return f"€{val:.2f}"
            
            # Create custom hover text for each trace
            revenue_hover = [format_hover_value(v) for v in metrics_over_time['Revenue']]
            costs_hover = [format_hover_value(v) for v in metrics_over_time['Total Costs']]
            profit_hover = [format_hover_value(v) for v in metrics_over_time['Profit']]
            
            # Add Revenue as a line
            fig_combined.add_trace(go.Scatter(
                x=metrics_over_time['Month'],
                y=metrics_over_time['Revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                text=revenue_hover,
                hovertemplate='Revenue: %{text}<extra></extra>'
            ))
            
            # Add Costs as a line
            fig_combined.add_trace(go.Scatter(
                x=metrics_over_time['Month'],
                y=metrics_over_time['Total Costs'],
                mode='lines+markers',
                name='Costs',
                line=dict(color='#d62728', width=2),
                marker=dict(size=6),
                text=costs_hover,
                hovertemplate='Costs: %{text}<extra></extra>'
            ))
            
            # Add Profit as bars with conditional coloring (green for positive, red for negative)
            profit_colors = ['#2ca02c' if p >= 0 else '#d62728' for p in metrics_over_time['Profit']]
            
            # For legend, use green or orange (user preference)
            # Since Plotly uses first color for legend, ensure it's green/orange
            # Use green (#2ca02c) as default, or orange (#ff7f0e) if all negative
            has_positive = any(p >= 0 for p in metrics_over_time['Profit'])
            legend_color = '#2ca02c' if has_positive else '#ff7f0e'  # Green or orange
            
            # If all negative, use orange for first bar so legend shows orange
            if not has_positive and len(profit_colors) > 0:
                profit_colors[0] = '#ff7f0e'  # Orange for legend when all negative
            
            fig_combined.add_trace(go.Bar(
                x=metrics_over_time['Month'],
                y=metrics_over_time['Profit'],
                name='Profit',
                marker=dict(color=profit_colors, opacity=0.6),
                text=profit_hover,
                hovertemplate='Profit: %{text}<extra></extra>'
            ))
            
            fig_combined.update_layout(
                title='Monthly Revenue, Costs & Profit',
                xaxis_title="Month",
                yaxis_title="Amount (€)",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_combined, use_container_width=True)
        
        with col2:
            fig_members_time = px.line(metrics_over_time, x='Month', y='Active Members',
                                      title='Active Members Evolution',
                                      markers=True,
                                      color_discrete_sequence=['#ff7f0e'])
            fig_members_time.update_layout(xaxis_title="Month", yaxis_title="Active Members")
            st.plotly_chart(fig_members_time, use_container_width=True)
        
        # Per-member revenue evolution
        st.subheader("📈 Revenue Per Active Member Over Time")
        metrics_over_time['Revenue Per Member'] = (metrics_over_time['Revenue'] / metrics_over_time['Active Members']).round(2)
        
        # Calculate per-member costs over time using metrics_over_time (already calculated)
        # Calculate per active member costs
        metrics_over_time['Total Cost Per Member'] = (
            (metrics_over_time['Variable Costs'] + metrics_over_time['Fixed Costs']) / 
            metrics_over_time['Active Members']
        ).round(2)
        metrics_over_time['Variable Cost Per Member'] = (
            metrics_over_time['Variable Costs'] / metrics_over_time['Active Members']
        ).round(2)
        
        # Replace inf and NaN with 0 (when Active Members is 0)
        metrics_over_time['Total Cost Per Member'] = metrics_over_time['Total Cost Per Member'].replace([np.inf, -np.inf], 0).fillna(0)
        metrics_over_time['Variable Cost Per Member'] = metrics_over_time['Variable Cost Per Member'].replace([np.inf, -np.inf], 0).fillna(0)
        metrics_over_time['Revenue Per Member'] = metrics_over_time['Revenue Per Member'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate shared y-axis range for both plots
        all_values = pd.concat([
            metrics_over_time['Revenue Per Member'],
            metrics_over_time['Total Cost Per Member'],
            metrics_over_time['Variable Cost Per Member']
        ])
        y_min = max(0, all_values.min() * 0.9)  # Start from 0 or slightly below min
        y_max = all_values.max() * 1.1  # Add 10% padding above max
        
        col1, col2 = st.columns(2)
        with col1:
            fig_rev_per_member = px.line(metrics_over_time, x='Month', y='Revenue Per Member',
                                         title='Revenue Per Active Member Evolution',
                                         markers=True,
                                         color_discrete_sequence=['#2ca02c'])
            fig_rev_per_member.update_layout(
                xaxis_title="Month", 
                yaxis_title="Revenue Per Member (€)",
                yaxis=dict(range=[y_min, y_max])
            )
            fig_rev_per_member.add_hline(y=revenue_per_member, line_dash="dash", 
                                        annotation_text=f"Current: €{revenue_per_member:.2f}",
                                        line_color="red")
            st.plotly_chart(fig_rev_per_member, use_container_width=True)
        
        with col2:
            fig_cost_per_member = px.line(metrics_over_time, x='Month', 
                                         y=['Variable Cost Per Member', 'Total Cost Per Member'],
                                         title='Cost Per Active Member Evolution',
                                         markers=True)
            fig_cost_per_member.update_layout(
                xaxis_title="Month", 
                yaxis_title="Cost Per Member (€)",
                yaxis=dict(range=[y_min, y_max])
            )
            fig_cost_per_member.add_hline(y=variable_cost_per_member, line_dash="dash", 
                                          annotation_text=f"Current Variable: €{variable_cost_per_member:.2f}",
                                          line_color="orange")
            st.plotly_chart(fig_cost_per_member, use_container_width=True)
        
        st.divider()
        
    
    # ============================================================================
    # TAB 2: Costs & CAC
    # ============================================================================
    with tab2:
        
        # CAC Analysis
        st.subheader("Customer Acquisition Cost (CAC) Analysis")
        
        # Temporal evolution of CAC
        if 'acquisition_cost_avg' in context.columns:
            cac_time = context[['month', 'acquisition_cost_avg']].copy()
            cac_time.columns = ['Month', 'CAC']
            cac_time = cac_time.sort_values('Month')
            
            fig_cac_time = px.line(cac_time, x='Month', y='CAC',
                                   title='CAC Evolution Over Time',
                                   markers=True)
            fig_cac_time.update_layout(xaxis_title="Month", yaxis_title="CAC (€)")
            st.plotly_chart(fig_cac_time, use_container_width=True)
        
        # Calculate CAC by channel - use acquisition_cost_avg from context based on signup month
        # All members who signed up in the same month have the same CAC
        if 'acquisition_cost_avg' in context.columns and len(active_data) > 0:
            # Get unique members and their signup dates
            members_with_signup = active_data[['member_id', 'signup_date', 'acquisition_channel']].drop_duplicates('member_id')
            members_with_signup['signup_date'] = pd.to_datetime(members_with_signup['signup_date'])
            members_with_signup['signup_month'] = members_with_signup['signup_date'].dt.to_period('M').astype(str)
            
            # Merge with context to get CAC for each signup month
            context_cac = context[['month', 'acquisition_cost_avg']].copy()
            context_cac['month'] = pd.to_datetime(context_cac['month']).dt.to_period('M').astype(str)
            context_cac.columns = ['signup_month', 'CAC']
            
            # Merge to get CAC for each member
            members_with_cac = members_with_signup.merge(context_cac, on='signup_month', how='left')
            members_with_cac['CAC'] = members_with_cac['CAC'].fillna(0)
            
            # Calculate CAC by channel (weighted average by number of members)
            cac_data = []
            for channel in active_data['acquisition_channel'].unique():
                channel_members_cac = members_with_cac[members_with_cac['acquisition_channel'] == channel]
                channel_members = active_data[active_data['acquisition_channel'] == channel]
                
                if len(channel_members_cac) > 0:
                    # Calculate weighted average CAC for the channel
                    avg_cac = channel_members_cac['CAC'].mean()
                else:
                    avg_cac = 0
                
                cac_data.append({
                    'Channel': channel.replace('_', ' ').title(),
                    'CAC': avg_cac,
                    'New Members': channel_members[channel_members['tenure_months'] <= 1]['member_id'].nunique(),
                    'Total Members': channel_members['member_id'].nunique()
                })
        
        if cac_data:
            cac_df = pd.DataFrame(cac_data)
            cac_df = cac_df.sort_values('CAC', ascending=False)
            cac_df['CAC'] = cac_df['CAC'].round(2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.space(100)
                st.dataframe(cac_df, use_container_width=True, hide_index=True)
            with col2:
                fig_cac = px.bar(cac_df, x='Channel', y='CAC',
                               title='CAC by Acquisition Channel',
                               color='CAC',
                               color_continuous_scale='Reds')
                st.plotly_chart(fig_cac, use_container_width=True)
        
        # Cohort CAC Payback Analysis
        st.subheader("Cohort CAC Payback Analysis")
        st.write("When each cohort reached different multiples of their CAC (by months since signup)")
        
        if 'acquisition_cost_avg' in context.columns:
            # Prepare data with filters
            cohort_data_filtered = members.copy()
            if selected_center != 'All':
                cohort_data_filtered = cohort_data_filtered[cohort_data_filtered['center'] == selected_center]
            if selected_plan != 'All':
                cohort_data_filtered = cohort_data_filtered[cohort_data_filtered['plan'] == selected_plan]
            
            # Create cohort based on signup_date
            cohort_data_filtered['signup_date'] = pd.to_datetime(cohort_data_filtered['signup_date'])
            cohort_data_filtered['cohort'] = cohort_data_filtered['signup_date'].dt.to_period('M')
            cohort_data_filtered['cohort_str'] = cohort_data_filtered['cohort'].astype(str)
            cohort_data_filtered['cohort_year'] = cohort_data_filtered['signup_date'].dt.year
            
            # Calculate cohort_age (months since signup)
            cohort_data_filtered['month_period'] = pd.to_datetime(cohort_data_filtered['month']).dt.to_period('M')
            cohort_data_filtered['cohort_age'] = (
                (cohort_data_filtered['month_period'] - cohort_data_filtered['cohort']).apply(lambda x: x.n)
            )
            
            # Get CAC for each signup month
            context_cac = context[['month', 'acquisition_cost_avg']].copy()
            context_cac['month'] = pd.to_datetime(context_cac['month']).dt.to_period('M').astype(str)
            context_cac.columns = ['cohort_str', 'CAC']
            
            # Calculate per-member cumulative margin over time
            # For each member, calculate their margin per month and cumulative margin
            member_margin = cohort_data_filtered.groupby(['member_id', 'cohort_str', 'cohort_age']).agg({
                'price_paid': 'sum',
                'cost_to_serve': 'sum'
            }).reset_index()
            member_margin['margin'] = member_margin['price_paid'] - member_margin['cost_to_serve']
            
            # Sort by member, cohort, and cohort_age
            member_margin = member_margin.sort_values(['member_id', 'cohort_str', 'cohort_age'])
            
            # Calculate cumulative margin for each member
            member_margin['cumulative_margin'] = member_margin.groupby('member_id')['margin'].cumsum()
            
            # Merge CAC with member data
            member_margin = member_margin.merge(context_cac, on='cohort_str', how='left')
            member_margin['CAC'] = member_margin['CAC'].fillna(0)
            
            # Add cohort year
            member_margin['cohort_year'] = pd.to_datetime(member_margin['cohort_str']).dt.year
            
            # Calculate when each member reached each multiple of their CAC
            multiples = [1, 3, 5, 10, 20]
            member_payback_results = []
            
            for member_id in member_margin['member_id'].unique():
                member_data = member_margin[member_margin['member_id'] == member_id].copy()
                member_cac = member_data['CAC'].iloc[0] if len(member_data) > 0 else 0
                cohort_str = member_data['cohort_str'].iloc[0] if len(member_data) > 0 else None
                cohort_year = member_data['cohort_year'].iloc[0] if len(member_data) > 0 else None
                
                if member_cac > 0:
                    for mult in multiples:
                        target_margin = member_cac * mult
                        # Find when cumulative margin reached target
                        reached = member_data[member_data['cumulative_margin'] >= target_margin]
                        if len(reached) > 0:
                            months_to_reach = reached['cohort_age'].iloc[0]
                            member_payback_results.append({
                                'member_id': member_id,
                                'Cohort': cohort_str,
                                'Year': cohort_year,
                                'Multiple': mult,
                                'Months to Reach': months_to_reach
                            })
            
            # Aggregate by cohort and year: calculate average months to reach each multiple
            if member_payback_results:
                member_payback_df = pd.DataFrame(member_payback_results)
                
                # Calculate average months to reach each multiple for each cohort
                payback_results = member_payback_df.groupby(['Cohort', 'Year', 'Multiple']).agg({
                    'Months to Reach': 'mean'
                }).reset_index()
                
                # Filter to only years 2022, 2023, 2024
                payback_df = payback_results[payback_results['Year'].isin([2022, 2023, 2024])]
                
                if len(payback_df) > 0:
                    # Aggregate by year: calculate average months to reach each multiple for each year
                    year_aggregated = payback_df.groupby(['Year', 'Multiple']).agg({
                        'Months to Reach': 'mean'
                    }).reset_index()
                    year_aggregated = year_aggregated.sort_values(['Year', 'Multiple'])
                    
                    # Create single line chart with all three years
                    fig = go.Figure()
                    
                    # Color mapping for years
                    year_colors = {2022: '#1f77b4', 2023: '#ff7f0e', 2024: '#2ca02c'}
                    year_names = {2022: '2022', 2023: '2023', 2024: '2024'}
                    
                    for year in sorted(year_aggregated['Year'].unique()):
                        year_data = year_aggregated[year_aggregated['Year'] == year].copy()
                        year_data = year_data.sort_values('Multiple')
                        
                        # Plot line for this year
                        fig.add_trace(go.Scatter(
                            x=year_data['Months to Reach'],
                            y=year_data['Multiple'],
                            mode='lines+markers',
                            name=f'Year {year}',
                            line=dict(color=year_colors.get(year, '#808080'), width=3),
                            marker=dict(size=10)
                        ))
                    
                    fig.update_layout(
                        title='CAC Payback by Multiple - Aggregated by Year (2022, 2023, 2024)',
                        xaxis_title='Months Since Signup',
                        yaxis_title='CAC Multiple',
                        yaxis=dict(
                            tickmode='array',
                            tickvals=[1, 3, 5, 10, 20],
                            ticktext=['1x', '3x', '5x', '10x', '20x']
                        ),
                        hovermode='closest',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for years 2022, 2023, or 2024.")
            else:
                st.warning("No cohort payback data available.")
        else:
            st.warning("CAC data not available in context.")
        
        st.divider()
        
    
    # ============================================================================
    # TAB 3: P&L Statement
    # ============================================================================
    with tab3:
        #st.header("📊 Margins")
        st.markdown(f"**Period:** {selected_month_str}")
        if selected_center != 'All':
            st.markdown(f"**Center:** {selected_center}")
        if selected_plan != 'All':
            st.markdown(f"**Plan:** {selected_plan}")
        
        # P&L Statement
        pnl_data = {
            'Item': [
                'Revenue',
                'Variable Costs (Cost to Serve)',
                'Contribution Margin',
                'Fixed Costs',
                'Net Result'
            ],
            'Amount': [
                total_revenue,
                -variable_costs,
                contribution_margin,
                -fixed_costs,
                net_result
            ]
        }
        pnl_df = pd.DataFrame(pnl_data)
        pnl_df['Amount'] = pnl_df['Amount'].round(2)
        pnl_df['% of Revenue'] = ((pnl_df['Amount'] / total_revenue * 100).round(2) if total_revenue > 0 else 0)
        
        # Format for display
        pnl_display = pnl_df.copy()
        pnl_display['Amount'] = pnl_display['Amount'].apply(lambda x: f"€{x:,.2f}")
        pnl_display['% of Revenue'] = pnl_display['% of Revenue'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(pnl_display, use_container_width=True, hide_index=True)
        
        # Temporal evolution of P&L
        #st.subheader("📈 P&L Evolution Over Time")
        
        # Calculate P&L over time
        pnl_time_data = []
        for month in sorted(temporal_data['month'].unique()):
            month_active = temporal_data[(temporal_data['month'] == month) & (temporal_data['status'] == 'active')]
            month_rev = month_active['price_paid'].sum()
            month_var = month_active['cost_to_serve'].sum() if 'cost_to_serve' in month_active.columns else 0
            month_fixed_row = context[context['month'] == month]
            month_fixed = month_fixed_row['monthly_fixed_costs'].iloc[0] if len(month_fixed_row) > 0 and 'monthly_fixed_costs' in month_fixed_row.columns else 0
            month_net = month_rev - month_var - month_fixed
            
            pnl_time_data.append({
                'Month': month,
                'Revenue': month_rev,
                'Variable Costs': month_var,
                'Fixed Costs': month_fixed,
                'Net Result': month_net
            })
        
        if pnl_time_data:
            pnl_time_df = pd.DataFrame(pnl_time_data)
            pnl_time_df = pnl_time_df.sort_values('Month')
        
        # P&L Visualization
        #col1 = st.columns(1)
        
        #with col1:
        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total", "relative", "total"],
            x=pnl_df['Item'],
            textposition="outside",
            text=[f"€{x:,.0f}" for x in pnl_df['Amount']],
            y=pnl_df['Amount'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(
            title="P&L Waterfall Chart",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        
        # Margin Analysis
        st.subheader("Margin Analysis")
        
        margin_data = {
            'Metric': [
                'Gross Margin %',
                'Contribution Margin %',
                'Net Margin %'
            ],
            'Value': [
                ((total_revenue - variable_costs) / total_revenue * 100) if total_revenue > 0 else 0,
                (contribution_margin / total_revenue * 100) if total_revenue > 0 else 0,
                (net_result / total_revenue * 100) if total_revenue > 0 else 0
            ]
        }
        margin_df = pd.DataFrame(margin_data)
        margin_df['Value'] = margin_df['Value'].round(2)
        margin_df['Value'] = margin_df['Value'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(margin_df, use_container_width=True, hide_index=True)
        
        # Temporal evolution of margins
        if pnl_time_data:
            margin_time = pd.DataFrame(pnl_time_data)
            margin_time = margin_time.sort_values('Month')
            margin_time['Gross Margin %'] = ((margin_time['Revenue'] - margin_time['Variable Costs']) / margin_time['Revenue'] * 100).round(2)
            margin_time['Contribution Margin %'] = ((margin_time['Revenue'] - margin_time['Variable Costs']) / margin_time['Revenue'] * 100).round(2)
            margin_time['Net Margin %'] = (margin_time['Net Result'] / margin_time['Revenue'] * 100).round(2)
            
            fig_margin_time = px.line(margin_time, x='Month', y=['Gross Margin %', 'Contribution Margin %', 'Net Margin %'],
                                     title='Margin Evolution Over Time',
                                     markers=True)
            fig_margin_time.update_layout(xaxis_title="Month", yaxis_title="Margin (%)")
            st.plotly_chart(fig_margin_time, use_container_width=True)
        

if __name__ == "__main__":
    main()



