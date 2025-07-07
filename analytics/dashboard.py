"""Interactive dashboard for AI Token Factory Economics Stack."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
import json
from pathlib import Path

from models.base_models import EconomicLayer
from models.neocloud_models import NeocloudProvider
from models.inference_models import InferenceProvider
from models.application_models import Application
from utils.formatters import format_currency, format_percentage, format_number
from analytics.economic_analyzer import EconomicAnalyzer


def create_dashboard(neoclouds: List[NeocloudProvider],
                    inference_providers: List[InferenceProvider],
                    applications: List[Application],
                    market_data: Dict[str, Any]) -> None:
    """Create interactive Streamlit dashboard for AI Token Factory.
    
    Args:
        neoclouds: List of neocloud providers
        inference_providers: List of inference providers
        applications: List of applications
        market_data: Market data and benchmarks
    """

    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .layer-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🏭 AI Token Factory")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Executive Summary", "Neoclouds", "Inference Providers", "Applications", "Full Stack Analysis", "Market Analysis"]
    )
    
    # Main content
    if page == "Executive Summary":
        show_executive_summary(neoclouds, inference_providers, applications, market_data)
    elif page == "Neoclouds":
        show_neocloud_analysis(neoclouds, market_data)
    elif page == "Inference Providers":
        show_inference_analysis(inference_providers, market_data)
    elif page == "Applications":
        show_application_analysis(applications, market_data)
    elif page == "Full Stack Analysis":
        show_full_stack_analysis(neoclouds, inference_providers, applications, market_data)
    elif page == "Market Analysis":
        show_market_analysis(market_data)


def show_executive_summary(neoclouds: List[NeocloudProvider],
                          inference_providers: List[InferenceProvider],
                          applications: List[Application],
                          market_data: Dict[str, Any]) -> None:
    """Show executive summary dashboard."""
    st.title("🏭 AI Token Factory Economics Stack")
    st.markdown("### Executive Summary")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate aggregate metrics
    total_neocloud_revenue = sum(nc.total_revenue() for nc in neoclouds)
    total_inference_revenue = sum(ip.total_revenue() for ip in inference_providers)
    total_app_revenue = sum(app.total_revenue() for app in applications)
    
    avg_neocloud_margin = sum(nc.gross_margin() for nc in neoclouds) / len(neoclouds) if neoclouds else 0
    avg_inference_margin = sum(ip.gross_margin() for ip in inference_providers) / len(inference_providers) if inference_providers else 0
    avg_app_margin = sum(app.gross_margin() for app in applications) / len(applications) if applications else 0
    
    with col1:
        st.metric(
            "Total Stack Revenue",
            format_currency(total_neocloud_revenue + total_inference_revenue + total_app_revenue),
            delta=format_percentage(12.5)
        )
    
    with col2:
        st.metric(
            "Avg Neocloud Margin",
            f"{avg_neocloud_margin:.2%}",
            delta=f"{0.021:.2%}"
        )
    
    with col3:
        st.metric(
            "Avg Inference Margin",
            f"{avg_inference_margin:.2%}",
            delta=f"{-0.013:.2%}"
        )
    
    with col4:
        st.metric(
            "Avg Application Margin",
            f"{avg_app_margin:.2%}",
            delta=f"{0.057:.2%}"
        )
    
    # Revenue flow visualization
    st.markdown("### Revenue Flow Through Stack")
    
    # Create Sankey diagram data
    fig = create_revenue_flow_chart(neoclouds, inference_providers, applications)
    st.plotly_chart(fig, use_container_width=True)
    
    # Layer comparison
    st.markdown("### Layer Performance Comparison")
    
    layer_data = {
        'Layer': ['Neoclouds', 'Inference Providers', 'Applications'],
        'Revenue': [total_neocloud_revenue, total_inference_revenue, total_app_revenue],
        'Avg Margin': [avg_neocloud_margin, avg_inference_margin, avg_app_margin],
        'Count': [len(neoclouds), len(inference_providers), len(applications)]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.bar(
            layer_data, x='Layer', y='Revenue',
            title='Revenue by Layer',
            color='Layer',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        fig_revenue.update_layout(showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_margin = px.bar(
            layer_data, x='Layer', y='Avg Margin',
            title='Average Gross Margin by Layer',
            color='Layer',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        fig_margin.update_layout(showlegend=False, yaxis_tickformat='.2%')
        st.plotly_chart(fig_margin, use_container_width=True)


def show_neocloud_analysis(neoclouds: List[NeocloudProvider], market_data: Dict[str, Any]) -> None:
    """Show neocloud analysis dashboard."""
    st.title("☁️ Neocloud Providers Analysis")
    
    if not neoclouds:
        st.warning("No neocloud data available.")
        return
    
    # Provider selection
    selected_provider = st.selectbox(
        "Select Neocloud Provider:",
        options=range(len(neoclouds)),
        format_func=lambda x: neoclouds[x].name
    )
    
    provider = neoclouds[selected_provider]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue", format_currency(provider.total_revenue()))
    with col2:
        st.metric("Gross Margin", f"{provider.gross_margin():.2%}")
    with col3:
        st.metric("Utilization", format_percentage(provider.calculate_utilization()))
    with col4:
        st.metric("Total GPUs", format_number(sum(cluster.gpu_count for cluster in provider.clusters)))
    
    # Cost breakdown
    st.markdown("### Cost Structure")
    costs = provider.total_cost()
    
    cost_data = {
        'Cost Type': [item.name for item in costs.items],
        'Amount': [item.amount for item in costs.items],
        'Category': [item.cost_type.value for item in costs.items]
    }
    
    fig_costs = px.pie(
        cost_data, values='Amount', names='Cost Type',
        title=f"Cost Breakdown - {provider.name}"
    )
    st.plotly_chart(fig_costs, use_container_width=True)
    
    # GPU cluster details
    st.markdown("### GPU Clusters")
    
    cluster_data = []
    for cluster in provider.clusters:
        cluster_data.append({
            'Cluster Name': cluster.cluster_id,
            'GPU Model': cluster.gpu_spec.model,
            'Total GPUs': cluster.gpu_count,
            'Memory per GPU': f"{cluster.gpu_spec.memory_gb}GB",
            'Utilization': f"{cluster.utilization_rate*100:.1f}%",
            'Power (kW)': cluster.total_power_consumption_kw
        })
    
    df_clusters = pd.DataFrame(cluster_data)
    st.dataframe(df_clusters, use_container_width=True)


def show_inference_analysis(inference_providers: List[InferenceProvider], market_data: Dict[str, Any]) -> None:
    """Show inference provider analysis dashboard."""
    st.title("🤖 Inference Providers Analysis")
    
    if not inference_providers:
        st.warning("No inference provider data available.")
        return
    
    # Provider selection
    selected_provider = st.selectbox(
        "Select Inference Provider:",
        options=range(len(inference_providers)),
        format_func=lambda x: inference_providers[x].name
    )
    
    provider = inference_providers[selected_provider]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue", format_currency(provider.total_revenue()))
    with col2:
        st.metric("Gross Margin", f"{provider.gross_margin():.2%}")
    with col3:
        st.metric("Tokens/Month", format_number(provider.usage_metrics.monthly_input_tokens + provider.usage_metrics.monthly_output_tokens))
    with col4:
        st.metric("Avg Response Time", f"{provider.usage_metrics.avg_response_time_ms}ms")
    
    # Model performance
    st.markdown("### Model Performance")
    
    model_data = []
    for model in provider.models:
        model_data.append({
            'Model': model.name,
            'Size': model.size.value,
            'Parameters': f"{model.parameters_b}B",
            'Context Length': f"{model.context_length:,}",
            'Input Price': f"${model.pricing.input_price_per_1k:.4f}/1K",
            'Output Price': f"${model.pricing.output_price_per_1k:.4f}/1K"
        })
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    # Token pricing comparison
    st.markdown("### Token Pricing Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_input = px.bar(
            df_models, x='Model', y=[float(p.split('$')[1].split('/')[0]) for p in df_models['Input Price']],
            title='Input Token Pricing',
            labels={'y': 'Price per 1K tokens ($)'}
        )
        st.plotly_chart(fig_input, use_container_width=True)
    
    with col2:
        fig_output = px.bar(
            df_models, x='Model', y=[float(p.split('$')[1].split('/')[0]) for p in df_models['Output Price']],
            title='Output Token Pricing',
            labels={'y': 'Price per 1K tokens ($)'}
        )
        st.plotly_chart(fig_output, use_container_width=True)


def show_application_analysis(applications: List[Application], market_data: Dict[str, Any]) -> None:
    """Show application analysis dashboard."""
    st.title("📱 Applications Analysis")
    
    if not applications:
        st.warning("No application data available.")
        return
    
    # Application selection
    selected_app = st.selectbox(
        "Select Application:",
        options=range(len(applications)),
        format_func=lambda x: applications[x].name
    )
    
    app = applications[selected_app]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue", format_currency(app.total_revenue()))
    with col2:
        st.metric("Gross Margin", f"{app.gross_margin():.2%}")
    with col3:
        st.metric("Monthly Users", format_number(app.user_metrics.monthly_active_users))
    with col4:
        st.metric("Revenue per User", format_currency(app.total_revenue() / app.user_metrics.monthly_active_users if app.user_metrics.monthly_active_users > 0 else 0))
    
    # User segments
    st.markdown("### User Segments")
    
    segment_data = []
    for segment in app.user_segments:
        segment_data.append({
            'Segment': segment.name,
            'Users': segment.user_count,
            'Monthly Revenue': segment.avg_monthly_revenue_per_user,
            'Token Usage': segment.avg_tokens_per_user_per_month,
            'Churn Rate': f"{segment.churn_rate:.1f}%"
        })
    
    df_segments = pd.DataFrame(segment_data)
    st.dataframe(df_segments, use_container_width=True)
    
    # Productivity impact
    st.markdown("### Productivity Impact")
    
    productivity = app.productivity_metrics
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Time Saved per User", f"{productivity.time_saved_hours_per_month:.1f} hours/month")
        st.metric("Productivity Increase", format_percentage(productivity.productivity_increase_percentage))
    
    with col2:
        st.metric("Task Completion Rate", format_percentage(productivity.task_completion_rate))
        st.metric("User Satisfaction", f"{productivity.user_satisfaction_score:.1f}/10")


def show_full_stack_analysis(neoclouds: List[NeocloudProvider],
                            inference_providers: List[InferenceProvider],
                            applications: List[Application],
                            market_data: Dict[str, Any]) -> None:
    """Show full stack analysis dashboard."""
    st.title("📊 Full Stack Analysis")
    
    # Economic analyzer
    analyzer = EconomicAnalyzer()
    
    # Value flow analysis
    st.markdown("### Value Flow Through Stack")
    
    # Create value flow visualization
    fig = create_value_flow_chart(neoclouds, inference_providers, applications)
    st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency metrics
    st.markdown("### Stack Efficiency Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Neoclouds")
        for nc in neoclouds[:3]:  # Show top 3
            efficiency = nc.gross_margin()
            st.metric(nc.name, f"{efficiency:.2%}")
    
    with col2:
        st.markdown("#### Inference Providers")
        for ip in inference_providers[:3]:  # Show top 3
            efficiency = ip.gross_margin()
            st.metric(ip.name, f"{efficiency:.2%}")
    
    with col3:
        st.markdown("#### Applications")
        for app in applications[:3]:  # Show top 3
            efficiency = app.gross_margin()
            st.metric(app.name, f"{efficiency:.2%}")
    
    # Optimization recommendations
    st.markdown("### Optimization Recommendations")
    
    # Generate recommendations for each layer
    all_layers = neoclouds + inference_providers + applications
    recommendations = analyzer._generate_layer_recommendations(all_layers)
    
    for rec in recommendations[:5]:  # Show top 5 recommendations
        with st.expander(f"🎯 {rec.title} (Priority: {rec.priority})"):
            st.write(f"**Layer:** {rec.layer}")
            st.write(f"**Description:** {rec.description}")
            st.write(f"**Expected Impact:** {rec.expected_impact}")
            st.write(f"**Implementation Effort:** {rec.implementation_effort}")


def show_market_analysis(market_data: Dict[str, Any]) -> None:
    """Show market analysis dashboard."""
    st.title("📈 Market Analysis")
    
    # Market size and growth
    st.markdown("### Market Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "AI Infrastructure Market",
            "$45.2B",
            delta="+23.4% YoY"
        )
    
    with col2:
        st.metric(
            "Inference API Market",
            "$12.8B",
            delta="+67.8% YoY"
        )
    
    with col3:
        st.metric(
            "AI Applications Market",
            "$89.6B",
            delta="+41.2% YoY"
        )
    
    # Industry benchmarks
    st.markdown("### Industry Benchmarks")
    
    benchmark_data = {
        'Metric': ['GPU Utilization', 'Inference Margin', 'App User Growth', 'Token Cost Efficiency'],
        'Industry Average': [75.2, 18.5, 156.7, 0.85],
        'Top Quartile': [89.1, 28.3, 245.2, 1.24],
        'Your Performance': [82.3, 22.1, 198.4, 1.08]
    }
    
    df_benchmarks = pd.DataFrame(benchmark_data)
    
    fig_benchmark = px.bar(
        df_benchmarks, x='Metric', y=['Industry Average', 'Top Quartile', 'Your Performance'],
        title='Performance vs Industry Benchmarks',
        barmode='group'
    )
    st.plotly_chart(fig_benchmark, use_container_width=True)


def create_revenue_flow_chart(neoclouds: List[NeocloudProvider],
                             inference_providers: List[InferenceProvider],
                             applications: List[Application]) -> go.Figure:
    """Create revenue flow Sankey diagram."""
    # Calculate revenues
    neocloud_revenue = sum(nc.total_revenue() for nc in neoclouds)
    inference_revenue = sum(ip.total_revenue() for ip in inference_providers)
    app_revenue = sum(app.total_revenue() for app in applications)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Raw Resources", "Neoclouds", "Inference Providers", "Applications", "End Users"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        ),
        link=dict(
            source=[0, 1, 2, 3],
            target=[1, 2, 3, 4],
            value=[neocloud_revenue, inference_revenue, app_revenue, app_revenue * 0.8]
        )
    )])
    
    fig.update_layout(
        title_text="Revenue Flow Through AI Token Factory Stack",
        font_size=12,
        height=400
    )
    
    return fig


def create_value_flow_chart(neoclouds: List[NeocloudProvider],
                           inference_providers: List[InferenceProvider],
                           applications: List[Application]) -> go.Figure:
    """Create value flow visualization."""
    # Create a more detailed flow chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Neoclouds', 'Inference Providers', 'Applications'],
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Calculate average efficiency for each layer
    avg_neocloud_eff = sum(nc.gross_margin() for nc in neoclouds) / len(neoclouds) if neoclouds else 0
    avg_inference_eff = sum(ip.gross_margin() for ip in inference_providers) / len(inference_providers) if inference_providers else 0
    avg_app_eff = sum(app.gross_margin() for app in applications) / len(applications) if applications else 0
    
    # Add efficiency gauges
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_neocloud_eff,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Efficiency"},
            gauge={'axis': {'range': [None, 2]},
                  'bar': {'color': "#1f77b4"},
                  'steps': [{'range': [0, 1], 'color': "lightgray"},
                           {'range': [1, 2], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 1.5}}
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_inference_eff,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Efficiency"},
            gauge={'axis': {'range': [None, 2]},
                  'bar': {'color': "#ff7f0e"},
                  'steps': [{'range': [0, 1], 'color': "lightgray"},
                           {'range': [1, 2], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 1.5}}
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_app_eff,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Efficiency"},
            gauge={'axis': {'range': [None, 2]},
                  'bar': {'color': "#2ca02c"},
                  'steps': [{'range': [0, 1], 'color': "lightgray"},
                           {'range': [1, 2], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 1.5}}
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Stack Layer Efficiency Analysis",
        height=400
    )
    
    return fig


if __name__ == "__main__":
    # This would be called from main.py or a separate dashboard script
    st.write("AI Token Factory Dashboard - Import this module to use create_dashboard()")